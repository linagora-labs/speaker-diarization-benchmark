#!/usr/bin/env python3

# This program takes a series of XML-formatted files and creates an
# RTTM v1.3 (https://catalog.ldc.upenn.edu/docs/LDC2004T12/RTTM-format-v13.pdf)
# output file.

# Typical invocation:
#
#   nite_xml_to_rttm.py ~/Downloads/ami_public_manual_1.6.2/words/ES2008a.*.words.xml | sort -n -k 4 > /tmp/ES2008a.rttm
#
import math
import sys
import xml.etree.ElementTree as ET

xml_trees = []


def print_rttm_line(start_time, end_time, speaker_num):
    print("SPEAKER test\t1\t{}\t{}\t<NA>\t<NA>\t{}\t<NA>\t<NA>".format(start_time, end_time - start_time,
                                                                            speaker_num))


def convert_xml_to_rttm():
    xml_file = xml_files[xml_file_index]
    
    tree = ET.parse(xml_file)
    
    root = tree.getroot()
    xml_trees.append(root)
    start_time = end_time = None
    for element in root:
        
        
        if element.tag == 'Turn'   and 'startTime' in element.attrib and 'endTime' in element.attrib and 'speaker' in element.attrib:
            
            start_time = (float(element.attrib['startTime'])) 
            #print(element.attrib['speaker'])
            end_time = (float(element.attrib['endTime'])) 
            print_rttm_line(start_time, end_time, speaker_num=element.attrib['speaker'])
            # if start_time is None:
                # start_time = float(element.attrib['startTime'])
                # print(start_time)
            # if end_time is None:
                #end_time = float(element.attrib['startTime'])  # yes, 'starttime'
                #print(end_time)
            #if math.isclose(end_time, float(element.attrib['startTime']), abs_tol=0.01):
                # collapse the two
                #end_time = float(element.attrib['endTime'])
            #else:
                #print_rttm_line(start_time, end_time, speaker_num=xml_file_index)
                #start_time = float(element.attrib['startTime'])
                #end_time = float(element.attrib['endTime'])
                #print('wajdi')
    #if not ((start_time is None) or (end_time is None)):
        #print_rttm_line(start_time, end_time, speaker_num=xml_file_index)


xml_files = sys.argv[1:]
for xml_file_index in range(len(xml_files)):
    convert_xml_to_rttm()
