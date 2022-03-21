import os,sys
import csv
import argparse

first_event = True

def read_csv_file(filename):
    csvfile =open(filename,newline='',encoding='utf-8-sig')
    dictreader = csv.DictReader(csvfile)
    return dictreader


def write_csv_file(filename):
    output = open(filename,'w', newline='' )
    writer = csv.writer(output, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    return writer


def do_loop(N,limits,repeat,par,out):
    if N==0:
        return
    else:
        for i in range(int((limits[N-1][2]-limits[N-1][1])/limits[N-1][3]+1)):
            par[N-1] = limits[N-1][1] + i*limits[N-1][3]
            do_loop(N-1,limits,repeat,par,out)

            global first_event
            if (first_event):
                line = [0]
                first_event = False
            else:
                line = [1]
            if (N==1):
                for j in range(len(par)):
                    line.append(limits[N-1+j][0])
                    line.append(par[j])

                #print (line)
                for k in range(repeat):
                    if k>0 and line[0]==0:
                        line[0] = 1
                    out.writerow(line)


def parse_args():
    parser = argparse.ArgumentParser(description='Make scan table')
    parser.add_argument("-i", "--input",
                    type=str,
                    required=True,
                    help="csv file with scan parameters")
    parser.add_argument("-o", "--output",
                    type=str,
                    required=True,
                    help="output scan table")
    parser.add_argument("-r", "--repeat",
                    type=int,
                    required=True,
                    help="number of repeated measurements")

    args    = parser.parse_args()
    one = args.input
    two = args.output
    three = args.repeat
    return one,two,three


def main():
    infile,outfile,repeat = parse_args()
    
    # read input table
    reader = read_csv_file(infile)

    # device parameters
    limits = []
    for row in reader:
        limits.append([row['Device'], float(row['min']), float(row['max']),float(row['step'])])
    print( limits ) 

    #number of devices to scan
    N = len(limits)                                                                                                                   

    # scan value holder
    par = [None]*N

    # initialize output table
    writer = write_csv_file(outfile)

    # fill output table
    do_loop(N, limits, repeat, par, writer)

if __name__ == "__main__":
    main()
