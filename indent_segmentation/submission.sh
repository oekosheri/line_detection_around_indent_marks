#!/usr/local_rwth/bin/bash

localDir=`pwd`
program=$localDir/training.py
run_file=$localDir/run_file.sh
submit_file=$localDir/sub_file.sh

rm post_output
echo -n >post_output

bs=16
epochs=200

# Grid submission for input arguments
for augment in  0 1
	do

	for lr in 0.001 0.0001
	do

	for count in 1 2
	do

	for gpu in 1 2
	do

		mkdir -p $localDir/"sub_"$augment$gpu$lr$count
		cd $localDir/"sub_"$augment$gpu$lr$count

		if [ $gpu = 1 ]
        then
            batch=$bs
            tasks=1
        else
            batch=$((${gpu}*bs))
            tasks=$gpu

        fi

	    	# adapting run file
    		sed -e "s|tag_program|${program}|g" ${run_file}  |\
        	sed -e "s/\<tag_epoch\>/${epochs}/g"| \
        	sed -e "s/\<tag_batch\>/${bs}/g"| \
			sed -e "s/\<tag_lr\>/${lr}/g"| \
			sed -e "s/\<tag_count\>/${count}/g"| \
        	sed -e "s/\<tag_aug\>/${augment}/g" > script.sh
			# submit
			sed -e "s/\<tag_task\>/${tasks}/g" ${submit_file} > submit.sh
			sbatch submit.sh

done

done

done

done
