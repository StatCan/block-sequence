# Usage

In order to sequence an area, you will require two input datasets. The first dataset is the set of geographies to break areas down in to. The second is a set of edges (each side of a road arc) that fall within the blocks.

Note: the field names presented below are for reference only. The field names are can be configured at run time.

## Block Representative Points

This file contains the child blocks denoted by a unique identifier (UID), the parent geography UID by which those blocks are grouped, and an x and y coordinate to be used as a representative point for each child geography. The file is formatted as follows:

|parent_block_uid|child_block_uid|rep_point_x|rep_point_y|
|-----------|-----------|----------------|-------|
|16261|289175|7453167.092938442|1160847.44|
|16261|289168|7452433.93880083|1161292.6385714286|
|16261|289172|7452705.200059065|1161144.8557142857|
|16242|289162|7447942.416303612|1159915.6357142858|
|16242|289165|7445582.40529888|1160806.642857143|

The exact order of the fields in the file is not important, as long as all of the fields are present. There should only ever be one instance of a child block, but a parent may be identified multiple times as a way to group the children together.

## Edge Definitions

The edge data file contains all the road arcs that are present within each of the child blocks defined in the block data file. If the edge data contains road arcs that do not fall within a block defined in the block data file they will not be sequenced.

The edge data file contains information formatted as follows:

|arc_uid|source|target|interior_flag|edge_uid|child_block_uid|parent_block_uid|leftrightflag|street_uid|
|-----|-----|-----|-----|-----|-----|-----|-----|------|
|1945279|4430|2035|0|1915196|289168|16261|L|97127|
|1944193|10585|4430|0|1915194|289168|16261|L|97127|
|1945269|2035|10585|0|1915190|289168|16261|L|97127|

## Running the Sequencer

The sequencing engine is exposed through a command line tool that was installed along with the Python package. This lets you process a particular set of data.

There are many options, but only the required arguments will be outlined here. For information about each argument, run the command with the `--help` switch to display all the options.

```
python3 -m blocksequence <path/to/edges.csv> <path/to/blocks.csv> [path/to/output.csv]
```

If the output path is not specified a file named `sequence_results.csv` will be created in the current directory.