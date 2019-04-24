-- block face list for all data

DROP VIEW IF EXISTS bf_list;
CREATE VIEW bf_list AS
SELECT ST_Y(ST_StartPoint(ST_LineMerge(r.geom))) || ',' || ST_X(ST_StartPoint(ST_LineMerge(r.geom))) as start_node, 
	ST_Y(ST_EndPoint(ST_LineMerge(r.geom))) || ',' || ST_X(ST_EndPoint(ST_LineMerge(r.geom))) as end_node, 
      r.ngd_uid, ST_Length(r.geom) AS length, 
      bf.bf_uid, bf.bb_uid, bf.arc_side, bf.lb_uid,
      lb.lu_uid, lb.s_flag, lb.lfs_uid
FROM ngd_al AS r 
LEFT JOIN bf AS bf ON bf.ngd_uid = r.ngd_uid
LEFT JOIN lb AS lb ON lb.lb_uid = bf.lb_uid;

-- parent geography adjacency table
DROP VIEW IF EXISTS adjacency;
CREATE VIEW adjacency AS
SELECT a.lu_uid as src, a.geom as src_geom,  b.lu_uid as adj, b.geom as adj_geom
FROM lu AS a, lu AS b 
WHERE ST_Touches(a.geom, b.geom)
AND a.lu_uid != b.lu_uid
ORDER BY a.lu_uid;


-- Specific to GeoPackage, PostGIS doesn't require these.
INSERT INTO gpkg_contents (table_name, identifier, data_type, srs_id) VALUES ('bf_list', 'bf_list', 'features', 300001)
INSERT INTO gpkg_geometry_columns (table_name, column_name, geometry_type_name, srs_id, z, m) VALUES ('bf_list', 'SHAPE', 'MULTILINESTRING', 300001, 0, 0)
