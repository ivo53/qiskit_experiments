import time
from qiskit.providers.jobstatus import JobStatus

SIZE_LIMIT = 250000
CIRC_LIMIT = 100
def run_jobs(circs, backend, duration, num_shots_per_exp=1024):
    num_exp = len(circs)
    num_shots_per_exp = 1024
    size = duration * num_exp * num_shots_per_exp

    job_ids = []
    values = []
    if size < SIZE_LIMIT * 1024 and num_exp <= CIRC_LIMIT:
        # print("size", size)
        # print("num_exp", num_exp)
        # print(circs)
        # print("len_circs", len(circs))
        # print("option 1")
        # return
        pi_job = backend.run(
            circs,
            shots=num_shots_per_exp
        )
        job_ids.append(pi_job.job_id())
        result = pi_job.result()
        for i in range(len(circs)):
            try:
                counts = result.get_counts(i)["1"]
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
        # print("size", size)
        # print("num_exp", num_exp)
        # print(parts)
        # import numpy as np
        # print((np.array([len(p) for p in parts]) != 100).any())
        # print("len_parts0", len(parts[0]))
        # print("len_parts", len(parts))
        # print("option 2")
        # return
        num_queued = 0
        for idx, circs_part in enumerate(parts):
            current_job = backend.run(
                circs_part,
                shots=num_shots_per_exp
            )
            num_queued += 1
            # wait if more than three jobs are queued
            if num_queued > 3:
                try:
                    first_job_status = jobs[-2].status() 
                    while first_job_status is JobStatus.QUEUED:
                        time.sleep(30)
                    while current_job.status() is JobStatus.CANCELLED:
                        current_job = backend.run(
                            circs_part,
                            shots=num_shots_per_exp
                        )
                        while current_job.status() is JobStatus.INITIALIZING or \
                              current_job.status() is JobStatus.VALIDATING:
                            time.sleep(30)
                    num_queued -= 1
                except IBMJobApiError as ex:
                    print("Error encountered: {}".format(ex))
            jobs.append(current_job)
            job_ids.append(current_job.job_id())
        
        for job, circs_part in zip(jobs, parts):
            result = job.result()
            for i in range(len(circs_part)):
                try:
                    counts = result.get_counts(i)["1"]
                except KeyError:
                    counts = 0
                values.append(counts / num_shots_per_exp)
    return values, job_ids