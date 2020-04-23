import csv
import os

targetdir = 'C:/Users/lenovo/Desktop/RA2_Completed_Srijith'
newname = 'C:/Users/lenovo/Desktop/renamed_files/'
with open('shared/BaseFilecsv.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        for filename in os.listdir(targetdir):
            if filename == row[3]:
                dir_path = 'C:/Users/lenovo/Desktop/renamed_files/'+ row[1] +'/'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    os.rename(targetdir + '/' + filename, dir_path + row[1] + row[2] + '.txt')
                else:
                    os.rename(targetdir + '/' + filename, dir_path + row[1] + row[2] + '.txt')
