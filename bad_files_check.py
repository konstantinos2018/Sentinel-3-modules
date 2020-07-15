# =============================================================================
# IMPORTS
# =============================================================================
# Python Modules
import os

# =============================================================================
# BEGIN
# =============================================================================
products = ['OLCI', 'SLSTR', 'SRAL']
filepath = r'C:\Users\vlachos\Desktop'.replace('\\','\\')

for prod in products:
	src = os.path.join(filepath, prod)
	lst_bad_zip = []
	size_thresh = 1 #593920 # expressed in kBytes

	print('Checking for bad zips...')
	for subdir, dirs, files in os.walk(src):
		for f in files:
			# get the size of the file in kBytes (division by 1024)
			s = os.path.getsize(os.path.join(subdir, f))/1024.0
			# Check if size is abnormally small which means that the file is bad
			if s < size_thresh:
				lst_bad_zip.append(f)
				# remove bad zip file
				os.remove(os.path.join(subdir, f))

	print('Checked')
	print('Writing bad files text file...')
    
	# Write the name of the bad zip files in a text file
	f_obj = open(os.path.join(src, 'Bad_zips_'+prod+'.txt'), 'w')
	for i in lst_bad_zip:
		f_obj.write('{0}\n'.format(i[:-4]))
	f_obj.close()

	print('File written')