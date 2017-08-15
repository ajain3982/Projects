
# coding: utf-8

# In[1]:


import os
import os.path
import sys
import time
import requests
import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
from collections import defaultdict
import bson
import pymongo
import phonenumbers


# ## Data Description

# The data I used was for south west delhi and can be downloaded using  [OverPass API](http://overpass-api.de/api/map?bbox=76.8799,28.4630,77.2541,28.6981). I will apply the techniques learned from Udacity's Data Wrangling with MongoDB course to explore, audit and clean this dataset then convert the xml to JSON. The area is shown below.

# In[1]:


from IPython.display import Image
Image(filename='img.png')


# # Part1: Auditing Data
# 
# Before we add the dataset into the database, we need to check and see if there is any potential problems.
# As the first step of auditing the dataset, let's find out how many kinds of elements there are and the occurrence of each element, to get the feeling on how much of which data we can expect to have in the map.

# Ok, then let's go ahead. 
# First of all,let's check different tags in the data.

# In[2]:


def count_tags(filename):
    tags = {}
    for event, elem in ET.iterparse(filename, events=('start', )):
        if elem.tag not in tags:
            tags[elem.tag] = 1
        else:
            tags[elem.tag] += 1
    return tags

start_time = time.time()
tags = count_tags("map")
sorted_by_occurrence = [(k, v) for (v, k) in sorted([(value, key) for (key, value) in tags.items()], reverse=True)]

print 'Element types and occurrence of data:\n'
pprint.pprint(sorted_by_occurrence)


# These are the different tags present above in the data.
# Let's check different keys  in the data.

# In[3]:


def count_keys(filename):
    keys = {}
    for event, elem in ET.iterparse(filename, events=('start', 'end')):
        if event == 'end':
            key = elem.attrib.get('k')
            if key:
                if key not in keys:
                    keys[key] = 1
                else:
                    keys[key] += 1
    return keys

start_time = time.time()
keys = count_keys("map")
sorted_by_occurrence = [(k, v) for (v, k) in sorted([(value, key) for (key, value) in keys.items()], reverse=True)]

print 'Keys and occurrence in data:\n'
pprint.pprint(sorted_by_occurrence)


# ###### The following function  displays  different keys with their occurrences in the xml data. It is used to check different values of a particular key for auditing purposes.

# In[24]:


def keys_with_val(filename,keyv):
    keys = {}
    for event, elem in ET.iterparse(filename, events=('start', 'end')):
        if event == 'end':
            key = elem.attrib.get('k')
            if key==keyv:
                v = elem.attrib.get('v')
                if v not in keys:
                    keys[v] = 1
                else:
                    keys[v]+=1
    return keys


# In[5]:


keys_with_val("map","addr:postcode")


# There are some postcodes that are not valid. For example  all postcodes starting with 12 and more than or less than 6 digits. We need to fix them, which will go into PART -2 of this notebook.

# In[10]:


keys_with_val("map","addr:country")


# All good here...!!!

# In[11]:


keys_with_val("map","addr:city")


# We need only city name, not state name. Hence stuff after ','  should be removed.

# In[52]:


keys_with_val("map","phone")


# # Part-2 : Fixing the above variables
# 
# #### Let's start by fixing pincodes. 

# We need to remove pincodes  and it's corresponding data because we don't require thatd data.

# In[13]:


def get_postcode(elem):
    if elem.tag in ['node', 'way', 'relation']:
        for tag in elem.iter():
            if tag.get('k') == 'addr:postcode':
                return True, tag.get('v')
        return False, None
    return False, None


def clean_postcode(filename, cleaned_filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    regex=re.compile('^12|^20|^02')
    
    for child in ['node', 'way', 'relation']:
        for elem in root.findall(child):
            has_postcode, postcode_value = get_postcode(elem)
            if has_postcode:
                if re.match(regex,postcode_value) or len(postcode_value)!=6:
                    root.remove(elem)
    
    return tree.write(cleaned_filename)

cleaned_postcode = 'cleaned_postcode.xml'
clean_postcode("map", cleaned_postcode)
    


# Now,let's check if th file still contains those postcodes.

# In[16]:


keys_with_val('cleaned_postcode.xml','addr:postcode')


# As we can see, the post code values are fixed.

# ##### Now let's change city names

# In[27]:


def clean_city(filename, cleaned_filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    
    for child in ['node', 'way', 'relation']:
        for elem in root.findall(child):
            if elem.tag in ['node', 'way', 'relation']:
                for tag in elem.iter():
                    if tag.get('k') == 'addr:city':
                        val = tag.get('v').split(",")[0]
                        tag.attrib['v'] = val 
    return tree.write(cleaned_filename)

cleaned_city = 'cleaned_city.xml'
clean_city("cleaned_postcode.xml", cleaned_city)


# In[30]:


keys_with_val("cleaned_city.xml","addr:city")


# #####  Let's fix some phone numbers
# 
# I am using google library phonenumbers for parsing and cleaning phone numbers .

# In[70]:


def proces_phone(number):
    try:
        parsed = phonenumbers.parse(number)
        return str(parsed.national_number)[0:-1]
    except:
        return number
        pass
    
def clean_phone(filename, cleaned_filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    
    for child in ['node', 'way', 'relation']:
        for elem in root.findall(child):
            if elem.tag in ['node', 'way', 'relation']:
                for tag in elem.iter():
                    if tag.get('k') == 'phone':
                        val = proces_phone(tag.get('v'))
                        tag.attrib['v'] = val 
    return tree.write(cleaned_filename)

cleaned_phone = 'cleaned_phone.xml'
clean_phone("cleaned_city.xml", cleaned_phone)


# In[71]:


keys_with_val("cleaned_phone.xml","phone")


# ## Part-3 : Importing into Database

# We've cleaned the phone numbers. The cleaned xml file is now cleaned_phone.xml. We'll convert this file to JSON and use it in MongoDB database.
# 

# In[72]:


lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]


# In[73]:


def shape_element(element):
    node = {}
    if element.tag == "node" or element.tag == "way":
        node['type'] = element.tag

        # Parse attributes
        for a in element.attrib:
            if a in CREATED:
                if 'created' not in node:
                    node['created'] = {}
                node['created'][a] = element.attrib[a]

            elif a in ['lat', 'lon']:
                if 'pos' not in node:
                    node['pos'] = [None, None]
                if a == 'lat':
                    node['pos'][0] = float(element.attrib[a])
                else:
                    node['pos'][1] = float(element.attrib[a])

            else:
                node[a] = element.attrib[a]

        # Iterate tag children
        for tag in element.iter("tag"):
            if not problemchars.search(tag.attrib['k']):
                # Tags with single colon
                if lower_colon.search(tag.attrib['k']):

                    # Single colon beginning with addr
                    if tag.attrib['k'].find('addr') == 0:
                        if 'address' not in node:
                            node['address'] = {}

                        sub_attr = tag.attrib['k'].split(':', 1)
                        node['address'][sub_attr[1]] = tag.attrib['v']

                    # All other single colons processed normally
                    else:
                        node[tag.attrib['k']] = tag.attrib['v']

                # Tags with no colon
                elif tag.attrib['k'].find(':') == -1:
                    node[tag.attrib['k']] = tag.attrib['v']

            # Iterate nd children
            for nd in element.iter("nd"):
                if 'node_refs' not in node:
                    node['node_refs'] = []
                node['node_refs'].append(nd.attrib['ref'])

        return node
    else:
        return None


# In[75]:


def process_map(file_in, pretty = False):
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data


# In[76]:


process_map('cleaned_phone.xml')


# In[ ]:




