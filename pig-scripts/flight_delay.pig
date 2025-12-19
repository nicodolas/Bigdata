-- flight_delay.pig (phiên bản cập nhật)
raw = LOAD 'hdfs://namenode:8020/input/200*.csv' USING PigStorage(',') 
    AS (
      f0:chararray,  -- Year
      f1:chararray,  -- Month
      f2:chararray,  -- DayofMonth
      f3:chararray,  -- DayOfWeek
      f4:chararray,  -- DepTime
      f5:chararray,  -- CRSDepTime
      f6:chararray,  -- ArrTime
      f7:chararray,  -- CRSArrTime
      f8:chararray,  -- UniqueCarrier
      f9:chararray,  -- FlightNum
      f10:chararray, -- TailNum
      f11:chararray, -- ActualElapsedTime
      f12:chararray, -- CRSElapsedTime
      f13:chararray, -- AirTime
      f14:chararray, -- ArrDelay
      f15:chararray, -- DepDelay
      f16:chararray, -- Origin
      f17:chararray, -- Dest
      f18:chararray, -- Distance
      f19:chararray, -- TaxiIn
      f20:chararray, -- TaxiOut
      f21:chararray, -- Cancelled
      f22:chararray, -- CancellationCode
      f23:chararray, -- Diverted
      f24:chararray, -- CarrierDelay
      f25:chararray, -- WeatherDelay
      f26:chararray, -- NASDelay
      f27:chararray, -- SecurityDelay
      f28:chararray  -- LateAircraftDelay
    );


no_header = FILTER raw BY (f0 IS NOT NULL) AND (TRIM(f0) != 'Year');


valid_rows = FILTER no_header BY
    TRIM(f1) MATCHES '^[0-9]+$' AND
    TRIM(f15) MATCHES '^-?[0-9]+(\\.[0-9]+)?$' AND
    TRIM(f14) MATCHES '^-?[0-9]+(\\.[0-9]+)?$';

-- Cast an toàn
parsed = FOREACH valid_rows GENERATE
    (int)TRIM(f1) AS Month,
    (double)TRIM(f15) AS DepDelay,
    (double)TRIM(f14) AS ArrDelay;


grp = GROUP parsed BY Month;
avg_delay = FOREACH grp GENERATE
    group AS Month,
    AVG(parsed.DepDelay) AS avg_dep_delay,
    AVG(parsed.ArrDelay) AS avg_arr_delay;


sorted_by_dep = ORDER avg_delay BY avg_dep_delay ASC;
sorted_by_arr = ORDER avg_delay BY avg_arr_delay ASC;
min_dep_month = LIMIT sorted_by_dep 1;
min_arr_month = LIMIT sorted_by_arr 1;


rounded = FOREACH avg_delay GENERATE Month,
    ROUND(avg_delay.avg_dep_delay, 2) AS avg_dep_delay_2,
    ROUND(avg_delay.avg_arr_delay, 2) AS avg_arr_delay_2;

DUMP min_dep_month;
DUMP min_arr_month;

STORE min_dep_month INTO 'hdfs://namenode:8020/output/min_dep_month' USING PigStorage(',');
STORE min_arr_month INTO 'hdfs://namenode:8020/output/min_arr_month' USING PigStorage(',');
STORE rounded INTO 'hdfs://namenode:8020/output/avg_delay_by_month' USING PigStorage(',');
