"""
engine/git_sync.py
===================
Git-based persistentie voor Streamlit Community Cloud.

Streamlit Cloud draait de app op een ephemeral filesystem: elke keer dat de
container herstart (redeploy, slaapstand na inactiviteit, crash) gaat elke
lokale wijziging aan data/*.csv en data/*.pkl verloren. Om vanaf mobiel
tijdens het toernooi uitslagen te kunnen invoeren zonder die data kwijt te
raken, wordt elke wijziging via git gecommit + gepusht naar de repo — de repo
zelf is dan de enige bron van waarheid, niet de container.

Authenticatie: fine-grained GitHub PAT in st.secrets["GITHUB_TOKEN"]
(scope: alleen Contents: Read & write op deze repo). Het token wordt nooit
weggeschreven naar .git/config — het wordt per commando meegegeven via
`git -c http.extraheader=...`, zodat er nergens een credential op disk
achterblijft die een volgende container-restart zou kunnen lekken.
"""

import base64
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).parent.parent
_GIT_TIMEOUT = 30  # seconden — voorkomt dat de UI blijft hangen bij netwerkproblemen


class GitSyncError(Exception):
    """Gegooid bij een mislukte pull/commit/push — moet nooit stil verdwijnen."""
    pass


def _get_token() -> str | None:
    """Leest GITHUB_TOKEN uit st.secrets. Geeft None als Streamlit-context ontbreekt
    (bijv. bij lokaal CLI-gebruik van auto_updater.py) of het secret niet is gezet."""
    try:
        import streamlit as st
        return st.secrets.get("GITHUB_TOKEN")
    except Exception:
        return None


def _auth_header(token: str) -> str:
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return f"http.extraheader=AUTHORIZATION: basic {basic}"


def _git(args: list[str], token: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["git"]
    if token:
        cmd += ["-c", _auth_header(token)]
    cmd += args
    return subprocess.run(
        cmd, cwd=REPO_DIR, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )


def _current_branch() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_DIR, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )
    branch = r.stdout.strip()
    return branch if branch and branch != "HEAD" else "main"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def git_pull() -> bool:
    """
    Haal de laatste staat op — bedoeld om vóór load_resources() te draaien bij
    het opstarten van de app, zodat een update vanaf een ander device (bijv.
    mobiel) wordt meegenomen na een container-restart.

    Faalt bewust "zacht" (print + return False) in plaats van een exception:
    dit draait bij app-opstart, vóór er een Streamlit-pagina is om een error
    op te tonen, en een schone checkout zonder GITHUB_TOKEN moet nog steeds
    kunnen opstarten met de data die al in de repo zit.
    """
    token = _get_token()
    branch = _current_branch()
    try:
        result = _git(["pull", "--ff-only", "origin", branch], token=token)
        if result.returncode != 0:
            print(f"⚠️  git pull mislukt ({branch}): {result.stderr.strip()}")
            return False
        return True
    except Exception as e:
        print(f"⚠️  git pull mislukt: {e}")
        return False


def git_commit_and_push(files: list[str], message: str) -> None:
    """
    Stage + commit + push de opgegeven bestanden (paden relatief aan de repo-
    root, bijv. "data/matches.csv") naar de huidige branch.

    Gooit GitSyncError bij elke fout (ontbrekend token, commit-fout, conflict
    bij gelijktijdige writes, netwerkfout) — de aanroeper (de Streamlit-UI)
    moet dit tonen via st.error(), nooit silent falen zodat een invoer
    onopgemerkt verloren gaat bij de volgende container-restart.
    """
    token = _get_token()
    if not token:
        raise GitSyncError(
            "GITHUB_TOKEN ontbreekt in st.secrets — de uitslag is lokaal "
            "weggeschreven maar NIET gepersisteerd. Bij de volgende "
            "container-restart gaat deze wijziging verloren. Voeg een "
            "fine-grained GitHub PAT toe (scope: Contents Read & write) "
            "als secret 'GITHUB_TOKEN'."
        )

    branch = _current_branch()

    add = _git(["add", *files], token=token)
    if add.returncode != 0:
        raise GitSyncError(f"git add mislukt: {add.stderr.strip()}")

    # Niets gewijzigd na add (bv. record_result werd twee keer met dezelfde
    # uitslag aangeroepen) — geen lege commit maken.
    diff = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=REPO_DIR, timeout=_GIT_TIMEOUT,
    )
    if diff.returncode == 0:
        return

    commit = subprocess.run(
        [
            "git",
            "-c", "user.email=wk2026-bot@streamlit.app",
            "-c", "user.name=WK2026 Predictor Bot",
            "commit", "-m", message,
        ],
        cwd=REPO_DIR, capture_output=True, text=True, timeout=_GIT_TIMEOUT,
    )
    if commit.returncode != 0:
        raise GitSyncError(f"git commit mislukt: {commit.stderr.strip()}")

    # Rebase op de laatste remote-staat vóór het pushen, zodat een gelijktijdige
    # invoer vanaf een ander device (bijv. iemand anders op vakantie) niet
    # botst — dit maakt een fast-forward push mogelijk in het gangbare geval.
    pull = _git(["pull", "--rebase", "origin", branch], token=token)
    if pull.returncode != 0:
        # Streamlit Cloud heeft geen shell-toegang — een vastgelopen rebase zou
        # deze container permanent onbruikbaar maken voor élke volgende sync
        # (ook voor niet-gerelateerde wedstrijden). Herstel daarom altijd naar
        # een schone staat: abort de rebase en reset naar de laatste remote-
        # commit. De net ingevoerde uitslag bestaat dan alleen nog lokaal in
        # deze (ephemeral) container en gaat dus verloren — vandaar de harde
        # foutmelding hieronder in plaats van een stille retry.
        subprocess.run(["git", "rebase", "--abort"], cwd=REPO_DIR, capture_output=True, timeout=_GIT_TIMEOUT)
        _git(["fetch", "origin", branch], token=token)
        subprocess.run(
            ["git", "reset", "--hard", f"origin/{branch}"],
            cwd=REPO_DIR, capture_output=True, timeout=_GIT_TIMEOUT,
        )
        raise GitSyncError(
            "git pull --rebase mislukt — vermoedelijk een conflict door een "
            "gelijktijdige update vanaf een ander device (bijv. model.pkl is "
            "binair en kan nooit automatisch samengevoegd worden). De zojuist "
            "ingevoerde uitslag is NIET gepusht en is teruggedraaid naar de "
            "laatste gesynchroniseerde staat, zodat deze container bruikbaar "
            "blijft voor volgende invoer. Ververs de pagina (haalt de laatste "
            "staat op) en voer de uitslag opnieuw in.\n"
            f"Details: {pull.stderr.strip()}"
        )

    push = _git(["push", "origin", branch], token=token)
    if push.returncode != 0:
        raise GitSyncError(f"git push mislukt: {push.stderr.strip()}")
