"""Long-running Prefect process (the `prefect-worker` service): schedules the flows.

- drift-monitor: every MONITOR_INTERVAL_MINUTES (default 5); retrains when drift is over threshold
- scheduled-retrain: weekly cron (RETRAIN_CRON, default Sunday 03:00 UTC)
"""

from __future__ import annotations

import os

from prefect import serve

from fraud.pipelines.flows import monitor_flow, retrain_flow


def main() -> None:
    interval = int(os.environ.get("MONITOR_INTERVAL_MINUTES", "5")) * 60
    monitor = monitor_flow.to_deployment(
        name="drift-monitor", interval=interval, parameters={"auto_retrain": True}
    )
    weekly = retrain_flow.to_deployment(
        name="scheduled-retrain",
        cron=os.environ.get("RETRAIN_CRON", "0 3 * * 0"),
        parameters={"trigger": "schedule"},
    )
    serve(monitor, weekly)


if __name__ == "__main__":
    main()
