PROJ=$1
REP_IDX="1"
#REP_IDX=$2

SIM_DIR="../workspace/simulate/${PROJ}"
EST_DIR="../workspace/estimate/${PROJ}"
mkdir -p ${EST_DIR}

EXT=("tre" "dat.nex" "dat.csv")
for ext in "${EXT[@]}"; do
    # set source/destination filenames
    SRC="${SIM_DIR}/sim.${REP_IDX}.${ext}"
    DST="${EST_DIR}/new.${REP_IDX}.${ext}"
    # if source file exists
    if [ -f "${SRC}" ]; then
        # copy to destination
        cp ${SRC} ${DST} 
    fi
done
