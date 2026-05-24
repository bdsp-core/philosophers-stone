FROM condaforge/miniforge3:24.11.3-0

WORKDIR /app

COPY environment.lock.yml /app/environment.lock.yml

RUN mamba env create -f /app/environment.lock.yml && conda clean -afy

ENV PATH="/opt/conda/envs/philosopher/bin:/opt/conda/bin:${PATH}"
ENV CONDA_DEFAULT_ENV="philosopher"
ENV MPLBACKEND="Agg"

COPY . /app

RUN python -m pip install --no-deps -e /app

ENTRYPOINT ["philosophers-stone"]
CMD ["--manifest_csv", "phi_manifest.csv", "--outdir", "phi_out_sample"]
