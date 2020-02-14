#!/bin/bash
echo "SET CURRENT DIRECTORY OF dhusget.sh"
cd C:\\Users\\vlachos\\Desktop\\script

echo "GIVE dhusget.sh EXECUTION PERMISSION"
chmod +x dhusget.sh

echo "INITIALIZATION OF VARIABLES"
# define star and end dates
start_d=2018-09-01
end_d=2019-03-31
# convert to date types
start_d=$(date -I -d "$start_d") || exit -1
end_d=$(date -I -d "$end_d") || exit -1

# step size
step=15
# initialize d
d="$start_d";
s=$(date -I -d "$d + $step day")
# command based on sensor you want to download
# SLSTR
# sensor_command="./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SLSTR -S ""$d""T00:00:00.000Z -E ""$s""T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SL_2_WST___ -o product -O C:\\Users\\vlachos\\Desktop\\SLSTR -F 'filename:*NT*' -l 100 -n 2 -N 10"
# OLCI
# sensor_command=./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i OLCI -S "$d"T00:00:00.000Z -E "$s"T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T OL_2_WFR___ -o product -O C:\\Users\\vlachos\\Desktop\\OLCI -F 'filename:*NT*' -l 100 -n 2 -N 10
# SRAL
# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SRAL -S "$d"T00:00:00.000Z -E "$s"T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SR_2_WAT___ -o product -O C:\\Users\\vlachos\\Desktop\\SRAL -F 'filename:*NT*' -l 100 -n 2 -N 10

echo "BEGIN DOWNLOAD..."
if [[ "$s" > "$end_d" ]]
then
	s=$end_d
	echo "NOW DOWNLOADING...PERIOD: $d $s"
	# Execute download command
	# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SLSTR -S "$d"T00:00:00.000Z -E "$s"T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SL_2_WST___ -o product -O C:\\Users\\vlachos\\Desktop\\SLSTR -F 'filename:*NT*' -l 100 -n 2 -N 10
	./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SRAL -S "$d"T00:00:00.000Z -E "$s"T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SR_2_WAT___ -o product -O C:\\Users\\vlachos\\Desktop\\SRAL -F 'filename:*NT*' -l 100 -n 2 -N 10
	echo "CHECK FOR BAD zip FILES"
	C:/Anaconda/python.exe c:/users/vlachos/desktop/bad_files_check.py
	echo "UNZIP FILES"
	C:/Anaconda/python.exe c:/users/vlachos/desktop/unzip_delete.py
	echo "SUBSEST netCDF FILES"
	C:/Anaconda/python.exe c:/users/vlachos/desktop/nc_subset_save.py
else
	while expr "$s" "<=" "$end_d" > /dev/null
	do
		echo "NOW DOWNLOADING...PERIOD: $d $s"
		# Execute download command
		# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SLSTR -S "$d"T00:00:00.000Z -E "$s"T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SL_2_WST___ -o product -O C:\\Users\\vlachos\\Desktop\\SLSTR -F 'filename:*NT*' -l 100 -n 2 -N 10
		./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SRAL -S "$d"T00:00:00.000Z -E "$s"T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SR_2_WAT___ -o product -O C:\\Users\\vlachos\\Desktop\\SRAL -F 'filename:*NT*' -l 100 -n 2 -N 10
		echo "CHECK FOR BAD zip FILES"
		C:/Anaconda/python.exe c:/users/vlachos/desktop/bad_files_check.py
		echo "UNZIP FILES"
		C:/Anaconda/python.exe c:/users/vlachos/desktop/unzip_delete.py
		echo "SUBSEST netCDF FILES"
		C:/Anaconda/python.exe c:/users/vlachos/desktop/nc_subset_save.py
		d=$(date -I -d "$d + $step day")
		s=$(date -I -d "$d + $step day")
		if [[ "$s" > "$end_d" ]]
		then
			s=$end_d
			echo "NOW DOWNLOADING...PERIOD: $d $s"
			# Execute download command
			# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SLSTR -S "$d"T00:00:00.000Z -E "$s"T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SL_2_WST___ -o product -O C:\\Users\\vlachos\\Desktop\\SLSTR -F 'filename:*NT*' -l 100 -n 2 -N 10
			./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SRAL -S "$d"T00:00:00.000Z -E "$s"T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SR_2_WAT___ -o product -O C:\\Users\\vlachos\\Desktop\\SRAL -F 'filename:*NT*' -l 100 -n 2 -N 10
			echo "CHECK FOR BAD zip FILES"
			C:/Anaconda/python.exe c:/users/vlachos/desktop/bad_files_check.py
			echo "UNZIP FILES"
			C:/Anaconda/python.exe c:/users/vlachos/desktop/unzip_delete.py
			echo "SUBSEST netCDF FILES"
			C:/Anaconda/python.exe c:/users/vlachos/desktop/nc_subset_save.py
			break
		fi
	done
fi

# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i OLCI -S "$start_d"T00:00:00.000Z -E 2017-11-32T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T OL_2_WFR___ -o product -O C:\\Users\\vlachos\\Desktop\\OLCI -F 'filename:*NT*' -l 100 -n 2 -N 10

# echo "execute OLCI 2"
# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i OLCI -S 2018-05-01T00:00:00.000Z -E 2018-05-31T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T OL_2_WFR___ -o product -O C:\\Users\\vlachos\\Desktop\\OLCI -F 'filename:*NT*' -l 100 -n 2 -N 10

# echo "execute OLCI 1"
# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i OLCI -S 2018-04-01T00:00:00.000Z -E 2018-04-30T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T OL_2_WFR___ -o product -O C:\\Users\\vlachos\\Desktop\\OLCI -F 'filename:*NT*' -l 100 -n 2 -N 10

# echo "execute OLCI 2"
# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i OLCI -S 2018-05-01T00:00:00.000Z -E 2018-03-31T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T OL_2_WFR___ -o product -O C:\\Users\\vlachos\\Desktop\\OLCI -F 'filename:*NT*' -l 100 -n 2 -N 10

# echo "execute SLSTR 1"
# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SLSTR -S 2018-04-01T00:00:00.000Z -E 2018-04-30T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SL_2_WST___ -o product -O C:\\Users\\vlachos\\Desktop\\SLSTR -F 'filename:*NT*' -l 100 -n 2 -N 10

# echo "execute SLSTR 2"
# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SLSTR -S 2018-05-01T00:00:00.000Z -E 2018-05-31T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SL_2_WST___ -o product -O C:\\Users\\vlachos\\Desktop\\SLSTR -F 'filename:*NT*' -l 100 -n 2 -N 10

# echo "execute SRAL 1"
# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SRAL -S 2018-04-01T00:00:00.000Z -E 2018-04-30T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SR_2_WAT___ -o product -O C:\\Users\\vlachos\\Desktop\\SRAL -F 'filename:*NT*' -l 100 -n 2 -N 10

# echo "execute SRAL 2"
# ./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i SRAL -S 2018-05-01T00:00:00.000Z -E 2018-05-31T00:00:00.000Z -c -80.00,31.00:-65.00,40.00 -T SR_2_WAT___ -o product -O C:\\Users\\vlachos\\Desktop\\SRAL -F 'filename:*NT*' -l 100 -n 2 -N 10

# echo "execute again"
# echo "./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i OLCI -S 2018-09-30T00:00:00.000Z -E 2018-10-30T00:00:00.000Z -c 0.70,36.00:4.90,41.80 -T OL_2_WFR___ -o product -O C:\\Users\\vlachos\\Desktop\\OLCI -F 'filename:*NT*' -l 100 -n 2 -N 10"

# echo "execute again"
# echo "./dhusget.sh -u konTinos18 -p myEumETSTATSentin3LE -m Sentinel-3 -i OLCI -S 2018-10-30T00:00:00.000Z -E 2018-11-30T00:00:00.000Z -c 0.70,36.00:4.90,41.80 -T OL_2_WFR___ -o product -O C:\\Users\\vlachos\\Desktop\\OLCI -F 'filename:*NT*' -l 100 -n 2 -N 10"

# echo "Execute python scripts"
# C:/Anaconda/python.exe c:/users/vlachos/desktop/bad_files_check.py

# C:/Anaconda/python.exe c:/users/vlachos/desktop/unzip_delete.py

# C:/Anaconda/python.exe c:/users/vlachos/desktop/nc_subset_save.py

# echo "Copy OLCI folder to POSTBOX"
# cp -R C:/Users/vlachos/Desktop/SRAL "N:/Deltabox/Postbox/Vlachos, Kostas/Gulf Stream"

# Leave terminal window open
$SHELL