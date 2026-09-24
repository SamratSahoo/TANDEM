"""TiPToP's runtime, as a recipe: three pinned trees, one patch, two checkpoints, one pixi environment.

Read by ``tandem.planners.runtime``, which does the fetching, patching and building; everything here
is a fact about TiPToP. A bump is an edit of the commits below -- nothing ships in the wheel any more,
so a new planner version no longer needs a new tandem release to reach a workstation, only a new pin.

The runtime keeps the directory layout of the monorepo these trees came from::

    <runtime>/
        tiptop/     cuTAMP/     curobo/
        vae/checkpoints/vae_full_v2.pt
        rnd/checkpoints/rnd_droid.pt
        env/                              the pixi environment; tiptop/.pixi points here

because three modules resolve default asset paths by walking up from ``__file__`` to what they take
to be a repo root:

    tiptop/tiptop/motion_planning.py                    parents[2]
    curobo/src/curobo/rollout/cost/vae_manifold_cost.py parents[5]
    curobo/src/curobo/rollout/cost/rnd_novelty_cost.py  parents[5]

Reproducing the layout makes all three resolve with no patching, and means an existing
``cfg/tamp/*.yml`` imports verbatim. cuTAMP keeps its capital T because tiptop's install-cutamp.sh
defaults to CUTAMP_DIR=../cuTAMP.

Imported to list planners, so it declares and does nothing: no file is read until an install runs.
"""

from __future__ import annotations

from pathlib import Path

from tandem.planners.base import SourcePin
from tandem.planners.runtime import Asset, BuildStep, PixiEnvironment, RuntimeRecipe, Source

_HERE = Path(__file__).resolve().parent
#: Unified diffs applied to the fetched trees, shipped as package data.
PATCHES = _HERE / "patches"
#: Files the runtime needs that no public repository has, shipped as package data.
ASSETS = _HERE / "assets"

# All three are the clean upstreams, with no phase-planning logic in any of them. That logic is
# tandem's own (src/tandem/planning), and tandem drives an unmodified planner through
# src/tandem/planners -- which is the whole point: a planner tandem has to fork is a planner tandem has
# to keep forking.
#
# tiptop and cuTAMP MUST move together, and nothing will catch it if they do not: both cuTAMP commits
# call themselves 0.0.5, so tiptop's check_cutamp_version passes either way. This tiptop imports
# cutamp.posture_prior and cutamp.particle_initialization.NoGraspsError and always passes the
# TAMPConfiguration fields transit_apex_*, grasp_center_cost, grasp_rank_conf_weight and
# require_m2t2_grasps; under the older cuTAMP that is an ImportError at warm, then a TypeError on
# every config. cuRobo did not move: SamratSahoo/curobo's main is still 3a90ff4, and it already has
# everything these two call on it (IKSolver.solve_batch(return_seeds=), the VAE cost's retiming).
# The sidecar is checked against these exact trees (tests/test_planners.py, tests/test_sidecar_legs.py,
# tests/test_tiptop_bump.py, run by CI with the pinned sources fetched), and so is the set of `tamp:`
# keys a profile may set (tamp_keys.py): move the pins and those checks together.
TIPTOP = Source(
    SourcePin(
        "tiptop", "https://github.com/SamratSahoo/tiptop.git", "1c6daf3f5d1ab822a0787c40ec0ed6b6caa472de"
    ),
    trim=(
        "docs/_static",  # 22 MB of screen recordings and screenshots
        "docs/blogs",
        ".github",
    ),
    patches=(
        # $TIPTOP_CALIBRATION: a profile owns its extrinsics instead of the shared runtime. The patch
        # says why at length; upstreaming it would leave this recipe with nothing to patch here.
        PATCHES / "0001-tiptop-config-from-env.patch",
        # There used to be a second one, making pyrealsense2 an extra. It is gone on purpose: tiptop's
        # pixi.lock records tiptop's own requires-dist, so editing that list makes the lock stale, and
        # `pixi install` then re-solves the whole PyPI side instead of installing what tiptop locked
        # (pixi 0.81, measured: "metadata for local package 'tiptop' has changed", then a re-solve;
        # `--locked` refuses a stale lock). The environment is linux-64 / Python 3.12 only, where the
        # locked pyrealsense2 wheel resolves, so the patch bought nothing there. Do not patch
        # pyproject.toml, pixi.toml or pixi.lock here; change them upstream, together.
    ),
    marker="pixi.toml",
    # tiptop downloads SAM-2's sam2.1_hiera_large.pt (~0.9 GB) into its own package, at
    # Path(tiptop.__file__).parent / ".cache" (utils.get_tiptop_cache_dir), the first time a session
    # warms (sam2_client). Inside the tree it would go with the tree on every pin bump, patch change or
    # --force, and the next warm-up would download it again -- inside WARM_TIMEOUT, with an operator
    # waiting at the arm, and not at all on a rig with no network. Kept under <runtime>/cache instead.
    persistent=("tiptop/.cache",),
)

CUTAMP = Source(
    SourcePin(
        "cuTAMP", "https://github.com/SamratSahoo/cuTAMP.git", "3a2e4d000339f7460f1989bde84d32b328ac92f9"
    ),
    trim=(
        "cutamp/robots/assets/yam_description",  # 11 MB; the bimanual YAM is out of scope
        "docs",
    ),
    marker="cutamp/__init__.py",
)

CUROBO = Source(
    SourcePin(
        "curobo", "https://github.com/SamratSahoo/curobo.git", "3a90ff49eee169d9636b2a679d98457a2592fb52"
    ),
    trim=(
        # 115 MB of robot meshes for arms this pipeline does not support. Franka and UR stay: cuTAMP's
        # fr3_robotiq and ur5e configs reference them.
        "src/curobo/content/assets/robot/techman",
        "src/curobo/content/assets/robot/iiwa_allegro_description",
        "src/curobo/content/assets/robot/jaco",
        "src/curobo/content/assets/robot/kinova",
        # An nvblox demo scene (a UR10 bins mesh). nvblox_torch is not installed and the world here
        # comes from the ZED point cloud, so nothing reads it.
        "src/curobo/content/assets/scene/nvblox",
        "images",
        "benchmark",
        "docker",
        ".github",
    ),
    marker="src/curobo",
)

RECIPE = RuntimeRecipe(
    planner="tiptop",
    title="TiPToP",
    sources=(TIPTOP, CUTAMP, CUROBO),
    # The DATAFARM checkpoints for the cuRobo fork's VAE-manifold and RND-novelty costs. Their source
    # repository is private, so these two small files (1.4 MB and 2.8 MB) are the only planner files
    # still shipped inside tandem. They go where those costs look by default (parents[5] above);
    # tandem also points VAE_MANIFOLD_CKPT / RND_NOVELTY_CKPT at them (render.py).
    assets=(
        Asset(ASSETS / "vae_full_v2.pt", "vae/checkpoints/vae_full_v2.pt"),
        Asset(ASSETS / "rnd_droid.pt", "rnd/checkpoints/rnd_droid.pt"),
    ),
    # tiptop's own manifest and lock: the environment is exactly the one tiptop pins.
    environment=PixiEnvironment(
        manifest="tiptop/pixi.toml",
        home="env",
        env={
            # An exported tree has no .git, so setuptools_scm cannot infer tiptop's version and its
            # editable install fails. tiptop's pixi.toml sets this too, under [activation.env]; belt
            # and braces, since `pixi install` builds the editable package before any activation.
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TIPTOP": "0.1.0",
        },
    ),
    steps=(
        # Build cuRobo's CUDA kernels, then install cuTAMP; tiptop's pixi task encodes that order
        # (cuTAMP imports cuRobo). The install scripts default to ../curobo and ../cuTAMP, which this
        # layout already matches; naming them makes a renamed directory a clear error instead of a
        # confusing one.
        BuildStep(
            name="planners",
            task="setup-planners",
            env={
                "CUROBO_DIR": "{source:curobo}",
                "CUTAMP_DIR": "{source:cuTAMP}",
                # cuRobo takes its version from setuptools_scm with no fallback, and an exported tree
                # has no SCM metadata -- that is the point of exporting it. Without this the editable
                # install fails with "unable to detect version ... make sure you're building from a
                # fully intact git repository" before the 5-20 minute kernel build even starts. The
                # commit rides along as a local version (0.0.0+g3a90ff4), so the installed package
                # says which sources it is. cuTAMP carries a static version and needs nothing.
                "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO": "{version:curobo}",
            },
            produces=("curobo/src/curobo/curobolib/*.so",),
            description="compiling cuRobo's CUDA kernels and installing cuTAMP — 5–20 minutes the first time",
            label="cuRobo kernels",
            done="compiled",
            todo="not compiled",
            problem="cuRobo's CUDA kernels have not been compiled",
        ),
    ),
    notes=(
        "Building the planner stack: torch, cuRobo (CUDA kernels), cuTAMP, tiptop.",
        "The first build fetches about 60 MB of sources, solves a CUDA environment, and compiles 5 CUDA "
        "extensions: 5–20 minutes.",
    ),
)

#: What an install builds the runtime from, for a catalog to show.
SOURCES = RECIPE.pins
