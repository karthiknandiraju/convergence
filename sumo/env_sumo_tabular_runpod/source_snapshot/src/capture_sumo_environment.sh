mkdir -p env

conda list --explicit > env/conda-explicit.txt
conda env export --no-builds > env/environment.yml
conda env export --from-history > env/environment-from-history.yml
conda list > env/conda-list.txt
python -m pip freeze > env/pip-freeze.txt

{
    echo "Captured: $(date --iso-8601=seconds)"
    echo "Conda environment: ${CONDA_DEFAULT_ENV:-unknown}"
    echo

    python --version
    conda --version

    echo
    sumo --version

    echo
    netconvert --version

    echo
    uname -a
} > env/system-info.txt