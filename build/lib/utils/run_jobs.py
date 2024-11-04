import time
from qiskit.providers.jobstatus import JobStatus
# from qiskit_ibm_provider.job import IBMJobApiError
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions
from qiskit_ibm_runtime.exceptions import (
    RuntimeJobFailureError, 
    RuntimeInvalidStateError, 
    IBMRuntimeError, 
    RuntimeJobMaxTimeoutError, 
    RuntimeJobTimeoutError,)
from qiskit_ibm_runtime.api.exceptions import RequestsApiError

SIZE_LIMIT = 250000
CIRC_LIMIT = 300
def run_jobs(circs, backend, optim_level, duration, num_shots_per_exp=1024, pm=None):
    num_exp = len(circs)
    size = duration * num_exp * num_shots_per_exp

    job_ids = []
    values = []
    if size < SIZE_LIMIT * 1024 and num_exp <= CIRC_LIMIT:
        # pi_job = backend.run(
        #     circs,
        #     shots=num_shots_per_exp
        # )
        options = SamplerOptions(optimization_level=optim_level)
        options.execution.shots = num_shots_per_exp
        pi_job = Sampler(backend, options=options).run(
            circs,
            shots=num_shots_per_exp
        )
        job_ids.append(pi_job.job_id())
        result = pi_job.result()
        for i in range(len(circs)):
            try:
                try:
                    counts = result[i].data.meas.get_counts()['1']
                except AttributeError as ae:
                    counts = result[i].data.c.get_counts()['1']
            except KeyError:
                counts = 0
            values.append(counts / num_shots_per_exp)
    else:
        max_size_part = int(min(CIRC_LIMIT, (SIZE_LIMIT * 1024) // (duration * num_shots_per_exp)))
        num_jobs = int(num_exp // max_size_part + bool(num_exp % max_size_part))
        extra = num_exp % num_jobs
        base_size_part = num_exp // num_jobs
        parts = [[] for _ in range(num_jobs)]
        for i in range(num_jobs):
            start = i * base_size_part + min(i, extra)
            end = (i + 1) * base_size_part + min(i + 1, extra)
            parts[i] = circs[start:end]
        jobs = []
        num_queued = 0
        for idx, circs_part in enumerate(parts):
            current_job = Sampler(backend).run(
                circs_part,
                shots=num_shots_per_exp
            )
            num_queued += 1
            # wait if more than three jobs are queued
            if num_queued >= 3:
                try:
                    while jobs[-2].status() is JobStatus.INITIALIZING or \
                            jobs[-2].status() is JobStatus.VALIDATING:
                        time.sleep(10)
                    while jobs[-2].status() is JobStatus.QUEUED or \
                          jobs[-2].status() is JobStatus.RUNNING:
                        time.sleep(10)
                    while current_job.status() is JobStatus.ERROR or\
                          current_job.status() is JobStatus.CANCELLED:
                        current_job = backend.run(
                            circs_part,
                            shots=num_shots_per_exp
                        )
                        while current_job.status() is JobStatus.INITIALIZING or \
                              current_job.status() is JobStatus.VALIDATING:
                            time.sleep(10)
                    num_queued -= 1
                except RuntimeJobFailureError or RuntimeInvalidStateError or IBMRuntimeError or RuntimeJobMaxTimeoutError or RuntimeJobTimeoutError or RequestsApiError as ex:
                    print("Error encountered: {}".format(ex))
            jobs.append(current_job)
            job_ids.append(current_job.job_id())
        
        for job, circs_part in zip(jobs, parts):
            result = job.result()
            for i in range(len(circs_part)):
                try:
                    try:
                        counts = result[i].data.meas.get_counts()['1']
                    except AttributeError as ae:
                        counts = result[i].data.c.get_counts()['1']
                except KeyError:
                    counts = 0
                values.append(counts / num_shots_per_exp)
    return values, job_ids