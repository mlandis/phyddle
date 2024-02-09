PROJ=$1
SRC_REP_IDX="1"
DST_REP_IDX="0"
#REP_IDX=$2

#SIM_DIR="../workspace/simulate/${PROJ}"
#EST_DIR="../workspace/estimate/${PROJ}"
SIM_DIR="../workspace/${PROJ}/simulate"
EST_DIR="../workspace/${PROJ}/estimate"
mkdir -p ${EST_DIR}

EXT=("tre" "dat.nex" "dat.csv" "labels.csv")
for ext in "${EXT[@]}"; do
    # set source/destination filenames
    SRC="${SIM_DIR}/sim.${SRC_REP_IDX}.${ext}"
    DST="${EST_DIR}/new.${DST_REP_IDX}.${ext}"
    # if source file exists
    if [ -f "${SRC}" ]; then
        # copy to destination
        cp ${SRC} ${DST} 
    fi
done
