CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'rep_secret';
SELECT pg_create_physical_replication_slot('replica_slot_1');