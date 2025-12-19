#!/usr/bin/env bash

# Export SPARK_HISTORY_OPTS để History Server đọc từ HDFS
export SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=hdfs://namenode:8020/spark-logs -Dspark.history.ui.port=18080"

/spark/sbin/start-master.sh
/spark/sbin/start-history-server.sh
