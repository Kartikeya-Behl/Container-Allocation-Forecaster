# Project Hand-over: Container CPU and Memory Forecasting

**Prepared:** 9 October 2026

> **Note:** Anything marked **UNVERIFIED** has not been confirmed and must be checked before you rely on it.

---

## 0. How to Use This Document

Give this whole document to the new assistant as your first message and say:

> Read this fully, repeat the plan back to me in ten lines, tell me what is unclear, and then help me with the next step in section 11.

The new assistant has no memory of the earlier chat, so this document is its only context.

---

## 1. What We Are Building

**Title:** A Lightweight Time-Series Forecasting Framework for Proactive CPU and Memory Allocation in Containerized Systems

**In plain words:** containers run with CPU and memory limits. If a limit is too low the container is throttled or killed for running out of memory, and if it is too high resources are wasted. Limits are usually fixed or change only after usage has already jumped. We predict each container's CPU and memory use a few minutes ahead and set the limits before the load arrives. We mainly use small, fast models and compare them with a small LSTM and with the usual static and reactive limits, to see whether the simple models are enough.

**Aim:** build and test a lightweight forecasting framework that predicts container CPU and memory usage and uses the predictions to set resource limits ahead of time.

**Who does what:** you do the main work. Your teammate is not very hands-on, so they run the overnight data collection on their own laptop (section 7).

**Time:** about 2 to 3 weeks of building.

---

## 2. Plan in Three Stages

| Stage | Week | Description | Status |
|-------|------|-------------|--------|
| Stage 1 | Week 1, **must finish** | Forecasting and replay evaluation on public Alibaba data. This is a complete project on its own. | Required |
| Stage 2 | Week 2 | Collect your own Docker usage data, use it as a second dataset, and run cross-dataset tests. | Required |
| Stage 3 | Week 3, only if Stages 1 and 2 are done | Live demo of static, reactive and proactive control on Docker. | Stretch |

**Effort in focused working days (about 3–4 hours each):**
- Stage 1: about 6–8 days
- Stage 2: about 3–4 more days (mostly waiting on overnight runs)
- Stage 3: about 4–6 more days

**Checkpoint:** by the end of week 1 there must be forecasting results on real Alibaba data. If Stage 1 is stuck, simplify it (fewer models, a smaller data slice, or the smaller 2017 trace) instead of starting Stage 2 early.

---

## 3. Goals (Each One Measurable)

1. **Collect and clean container usage data:**
   - CPU and memory series from the Alibaba microservices trace (3–4 hours at 60-second intervals, keeping at least 200 containers with complete data)
   - From our own Docker containers under scripted load (about 8 hours sampled every 5 seconds, with five load patterns and a fixed random seed)

2. **Build the forecasting models and test their accuracy:**
   - Naive last value, moving average, exponential smoothing or ARIMA, Random Forest, XGBoost and, if time allows, a small LSTM
   - Test one step ahead and about five minutes ahead on a time-ordered 70/15/15 split with MAE and RMSE
   - Use a statistical test before claiming one model is better

3. **Compare ways of setting limits by replaying the forecasts as limits:**
   - Static, reactive and proactive, judged by violation rate, under-provisioning and average headroom
   - Also check how the number of past values (lags) and the size of the safety margin change the results

4. **Measure the cost and check that results carry over:**
   - Training time, prediction time and model size for every model on the same machine
   - Train on one dataset and test on the other (both directions) after resampling both to the same interval

5. **Stretch:** run static, reactive and proactive control live on a Docker container under changing load, about five runs each, tracking throttling, OOM kills and wasted headroom

---

## 4. Technical Design

**Pipeline:** raw data → cleaning → resampling → lag and window features → model → forecast → allocation rule → replay simulation → metrics

**Split:** chronological only (first 70% train, next 15% validation, last 15% test), never shuffled. Decide between per-container models and one global model, and fix random seeds.

**Horizon:** next step, and about 5 minutes ahead.

**Proactive rule:** `limit = max(forecast over horizon) × (1 + safety_margin)`, with the margin tuned on the validation set.

**Static baseline:** a fixed limit from a training-set percentile (e.g. p95 or p99).

**Reactive baseline:** `limit = last observed usage + margin`, so it always lags the load.

**Metrics:**
- MAE, RMSE, sMAPE (careful with near-zero values)
- Under-provisioning rate and size
- Over-provisioning (average headroom)
- Prediction latency, training time, model size

**Statistical test:** e.g. Wilcoxon across containers, before claiming one method beats another.

**Ablations:** number of lags, with and without the margin, with and without extra features.

> **Note:** Alibaba data has no real container limits, so the replay uses hypothetical limits on normalised usage.

---

## 5. Public Data: Alibaba Cluster Trace Program

**Repository:** https://github.com/alibaba/clusterdata

The README says the traces may be used for research or study purposes. There is no formal licence text, so credit Alibaba and do not claim ownership.

| Trace | What it has | Size and interval | Verdict |
|-------|-------------|-------------------|---------|
| microservices-v2022 | 13 days in 2022, container CPU and memory utilisation (470,000+ containers), downloadable by day and hour | Container resource table about 700 MiB compressed per hour, 60 s interval, all 13 days about 2 TB | **RECOMMENDED**, take 3–4 contiguous hours |
| microservices-v2021 | 12 hours, 90,000+ containers | 30 s interval, resource table listed as about 16 GiB (**UNVERIFIED** whether per hour or total) | Option, check size first |
| cluster-trace-v2017 | about 1,300 machines, about 12 hours, container_usage.csv with CPU and memory as percent of requested | about 300 s interval, about 144 points per series, download after a short survey, size not checked | Quick-start or backup |
| cluster-trace-v2018 | 8 days, about 4,000 machines | container usage alone 28 GB compressed, 49 GB in total, about 280 GB extracted | **NOT suitable** |

**Notes:**
- Utilisation values are normalised, so results are in relative units.
- The v2022 per-hour sizes of the other tables are large (call graphs about 4 GiB per hour, another metrics table about 3 GiB), so `fetchData.sh` may download everything (about 7.7 GB per hour). **UNVERIFIED:** read the script first and edit it to fetch only the container resource table (MSResource).
- Usage: `bash fetchData.sh start_date=0d0 end_date=1d1`, where the format is `DAYdHOUR` and the range includes the start and excludes the end.
- Extracted size is unknown (the README lists compressed sizes only). Download one hour and measure it.
- Expected columns in MSResource: `timestamp`, `msname`, `msinstanceid`, `cpu_utilization`, `memory_utilization` (confirm in the README).
- Four hours at 60 s gives about 240 points per container. That is fine for simple models and Random Forest, and thin for an LSTM.

**Low-disk workflow:** download one hour, extract it, keep only a few hundred containers with complete regular series, save the result as a small CSV or Parquet file (likely tens of MB), delete the raw files, and repeat. Peak disk use is one hour of raw data. Colab or a friend's drive can do the heavy step if your disk is too small.

**Go or no-go test (do this first, about an hour):** download one hour of v2022 MSResource, load it in pandas and plot one container's CPU. If it works, Stage 1 is safe. If it fails, switch to v2017.

---

## 6. Environment

- Python 3.10 or newer with pandas, numpy, matplotlib, scikit-learn, statsmodels, and optionally xgboost.
- Add PyTorch or TensorFlow only if you do the LSTM.
- Docker Desktop (Windows or macOS) or Docker Engine (Linux), plus the Docker Python SDK (`pip install docker`).
- No separate Linux VM is needed, because Docker Desktop runs containers in its own hidden Linux VM.
- Jupyter or VS Code, plus Git and GitHub from day one.
- **Hardware guess:** 8 GB RAM minimum, 16 GB comfortable. Check Docker Desktop's resource cap.

**Early checks on both laptops:**
- Virtualisation enabled in the BIOS (VT-x or AMD-V)
- `docker run hello-world` works
- `docker stats` shows CPU and memory for a running container

> Docker reports CPU relative to one core, so values can exceed 100% for multi-core containers: record the CPU limit and normalise.

---

## 7. Stage 2: Own Docker Dataset

**Collector:** Python Docker SDK (`container.stats` with `stream=False`) or `docker stats --no-stream`, sampling every 1–5 seconds (5 s for 8 hours gives about 5,760 points per metric).

**CSV columns:** `timestamp (UTC)`, `container_name`, `cpu_percent`, `cpu_limit_cores`, `memory_usage_bytes`, `memory_limit_bytes`, optionally network and block I/O.

**Load generator:** your own small Python script inside a container that burns CPU and allocates memory on a schedule: steady, sudden spikes, slow ramps, periodic waves, random. Use a fixed random seed and log the pattern schedule so you can describe the data.

**Procedure:**
1. Write and test on your laptop.
2. Run a 10-minute test on the teammate's laptop and check the CSV before any long run.
3. Then do an overnight run of about 8 hours, or several 2–3 hour sessions.
4. Plug in power, turn off sleep and auto-updates, keep the machine idle, and check Docker Desktop's resource limits.
5. Record CPU, RAM, OS and Docker version.
6. Give the teammate one command and a short README. They send the CSV by Drive or WhatsApp.

Before cross-dataset tests, resample both datasets to the same interval (e.g. 60 s). Train on Alibaba and test on your data, and the reverse.

---

## 8. Stage 3: Live Control Demo (Stretch)

**Loop** every few seconds: read recent usage, forecast the next interval, and set limits with `docker update` (CPU limit plus memory limit; memory and swap limits may need updating together) while a test app runs under a changing load pattern.

**Compare three controllers:** static, reactive, proactive.

**Measure:** throttling and OOM kills (`OOMKilled` appears in `docker inspect` output, and CPU throttling counters live in the container's cgroup statistics, so check where to read them on your OS), application slowdown, and wasted headroom.

**Scale:** one container, one or two load patterns, three controllers, about 5 repeats, so about 15 runs of 10–15 minutes (3–4 hours in total, scripted to run unattended). Run all 15 on one machine under identical conditions. Do the live demo on the laptop you will present from. Offline results stay your main evidence.

---

## 9. Limits to Keep in Mind When Reading Results

- Normalised utilisation
- Coarse sampling (30–300 s)
- Short series
- Hypothetical limits in the replay
- No real application response time
- Synthetic workloads on a laptop for your own data
- Docker Desktop's Linux VM on Windows and macOS

---

## 10. Working Rules

- You run the code and Docker, and paste outputs and errors back. The assistant cannot run Docker or download the data for you.
- Fix random seeds, keep logs, settings and results for every run, and commit to GitHub as you go.
- Never invent or tune numbers to look better. Report whatever the experiments show.
- Build in small pieces and ask the assistant to explain each one, so you can explain the code yourself.

---

## 11. Open Information and First Steps

**Tell the new assistant:** your and your teammate's operating systems, your free disk space, and whether Docker is already installed.

**Suggested first requests, in order:**

1. Write a script that reads `fetchData.sh` (I will paste it), downloads only one hour of the container resource table, filters to containers with complete 60 s series, saves a small file and deletes the raw data. My OS is __ and my free disk is __ GB.
2. Build the Stage 1 pipeline: chronological split, naive and moving-average baselines, ARIMA or exponential smoothing, Random Forest and XGBoost with lag features, MAE and RMSE, and timing.
3. Build the replay simulator for static, reactive and proactive limits with violation, under-provisioning and headroom metrics.
4. Write the Docker stats collector, the load generator and a one-command README for my teammate.
5. Add the cross-dataset tests and the ablations, then, if time allows, the Stage 3 controller.

---

## 12. Links

- https://github.com/alibaba/clusterdata (trace list and terms of use)
- https://github.com/alibaba/clusterdata/tree/master/cluster-trace-microservices-v2022
- https://github.com/alibaba/clusterdata/blob/master/cluster-trace-microservices-v2021/README.md
- https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2017/trace_201708.md
- https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md
