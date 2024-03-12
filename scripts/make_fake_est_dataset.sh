PROJ=$1
SRC_REP_IDX="1"
DST_REP_IDX="0"
#REP_IDX=$2

SIM_DIR="./workspace/${PROJ}/simulate"
EMP_DIR="./workspace/${PROJ}/empirical"
mkdir -p ${EMP_DIR}

EXT=("tre" "dat.nex" "dat.csv" "labels.csv")
for ext in "${EXT[@]}"; do
    # set source/destination filenames
    SRC="${SIM_DIR}/out.${SRC_REP_IDX}.${ext}"
    DST="${EMP_DIR}/out.${DST_REP_IDX}.${ext}"
    # if source file exists
    if [ -f "${SRC}" ]; then
        # copy to destination
        cp ${SRC} ${DST} 
    fi
done
