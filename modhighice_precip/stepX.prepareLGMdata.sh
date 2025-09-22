###for experiment tdab, the data we have is not tempareture anomaly but the absolute temperature
###so we need to calculate the temperature anomaly by subtracting the preindustrial temperature from the LGM temperature
###the preindustrial temperature is the 1st simulation of tdab

#### training data for the output
#!/bin/bash

# Set output folder
output_folder="/Users/bo20541/Library/CloudStorage/OneDrive-UniversityofBristol/TONIC-Oligocene/Emulator_Charlie/Emulator/2015_Bristol_5D_v001/orig/Output/Singarayer + Valdes 2010/"

# Define file paths
output_path5="dT500_precip_mm_srf_ann_tdab28k.nc"
output_path6="dT500_precip_mm_srf_ann_tdab80k.nc"
output_path7="dT500_precip_mm_srf_ann_tdab120k.nc"

# Define experiment IDs and variable lists
output_expID=("tdab" "tdab" "tdab")

output_list5=("tdaba" "tdabb" "tdabc" "tdabd" "tdabe" "tdabf" "tdabg" "tdabh" "tdabi" "tdabj" "tdabk" "tdabl" "tdabm" "tdabn" "tdabo" "tdabp" "tdabq" "tdabr" "tdabs" "tdabt" "tdabu" "tdabv" "tdabw" "tdabx" "tdaby" "tdabz")
output_list6=("tdabA" "tdabB" "tdabC" "tdabD" "tdabE" "tdabF" "tdabG" "tdabH" "tdabI" "tdabJ" "tdabK" "tdabL" "tdabM" "tdabN" "tdabO" "tdabP" "tdabQ" "tdabR" "tdabS" "tdabT" "tdabU" "tdabV" "tdabW" "tdabX" "tdabY" "tdabZ")
output_list7=("tdab0" "tdab1" "tdab2" "tdab3" "tdab4" "tdab5" "tdab6" "tdab7" "tdab8" "tdab9")

output_paths=("$output_path5" "$output_path6" "$output_path7")
output_lists=("${output_list5[@]}" "${output_list6[@]}" "${output_list7[@]}")

output_all="dT500_precip_mm_srf_ann_tdab_all.nc"
##use ncks -A to append files
for output_path in "${output_paths[@]}"; 
do
    ncks -A "${output_folder}${output_path}" "${output_all}"
done

ncap2 -s "tdaba_orig=tdaba" "${output_all}" -O "${output_all}"

# Loop through each output path and corresponding variable list
for var in "${output_lists[@]}"; 
do
echo "Processing ${var}"
ncap2 -s "${var}=${var}-tdaba_orig" "${output_all}" -O "${output_all}"
done
# Select the vars of output_list5 from all into 28k
ncks -v "$(IFS=,; echo "${output_list5[*]}")" "${output_all}" -O ${output_folder}dT500_precip_mm_srf_ann_tdab28k_anomaly.nc
ncks -v "$(IFS=,; echo "${output_list6[*]}")" "${output_all}" -O ${output_folder}dT500_precip_mm_srf_ann_tdab80k_anomaly.nc
ncks -v "$(IFS=,; echo "${output_list7[*]}")" "${output_all}" -O ${output_folder}dT500_precip_mm_srf_ann_tdab120k_anomaly.nc
