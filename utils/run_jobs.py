SIZE_LIMIT = 258144
CIRC_LIMIT = 100
def run_jobs(circs, backend, duration, num_shots_per_exp=1024):
    num_exp = len(circs)
    size = duration * num_exp

    job_ids = []
    values = []
    if size < SIZE_LIMIT and num_exp <= CIRC_LIMIT:
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
        num_jobs = size // SIZE_LIMIT + 1
        size_part = num_exp // num_jobs
        if size_part > CIRC_LIMIT:
            size_part = CIRC_LIMIT
            num_jobs = num_exp // size_part + int(num_exp % size_part > 0)
        extra = num_exp % num_jobs
        parts = [[] for _ in range(num_jobs)]
        for i in range(num_jobs):
            start = i * size_part + min(i, extra)
            end = (i + 1) * size_part + min(i + 1, extra)
            parts[i] = circs[start:end]

        jobs = []
        # print([len(p) for p in parts])
        # return
        for circs_part in parts:
            current_job = backend.run(
                circs_part,
                shots=num_shots_per_exp
            )
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