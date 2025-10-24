import pickle
import re
import pandas as pd

# mapping ctv3 to icd10
ctv_icd_map = pd.read_excel("../../../data/all_lkps_maps_v3.xlsx", 
                            engine='openpyxl', sheet_name = ["read_ctv3_icd10"])["read_ctv3_icd10"]
ctv_icd_map = ctv_icd_map[~ctv_icd_map["mapping_status"].isin(["R","U"])]
ctv_icd_map = ctv_icd_map[ctv_icd_map["refine_flag"]!="M"]
ctv_icd_map = ctv_icd_map[ctv_icd_map["element_num"]==0] 
ctv_icd_map["icd10_code"] = ctv_icd_map["icd10_code"].apply(lambda x: re.sub('[a-zA-Z]+$', '', x))
read2_read3 = pd.read_excel("../../../data/all_lkps_maps_v3.xlsx", 
                            engine='openpyxl', sheet_name = ["read_v2_read_ctv3"])["read_v2_read_ctv3"]

# t=pd.read_csv("../../../data/gp_clinical.txt", delimiter='\t', chunksize=10000)#Your own file
t=pd.read_csv("../../../data/gp_clinical.csv", chunksize=10000)#Your own file


# adjusting icd10
phe_icd = pd.read_csv("../../../data/phecode_icd10.csv")
phe_icd_code = phe_icd.ALT_CODE.tolist()
for i,code in enumerate(phe_icd_code):
    if code[-1]=='X':
        phe_icd_code[i]=code[:-1]
        # print(code,code[:-1])
for i in range(len(ctv_icd_map)):
    if len(ctv_icd_map.iloc[i,1])>=4 and ctv_icd_map.iloc[i,1] not in phe_icd_code:
        ctv_icd_map.iloc[i,1] = ctv_icd_map.iloc[i,1][:-1] 
        

# two-step mapping
data_chunks = []
# reader = pd.read_csv("../../../data/gp_clinical.txt", delimiter='\t', chunksize=10000)
reader = pd.read_csv("../../../data/gp_clinical.csv", chunksize=10000)
for id,chunk in enumerate(reader):
    chunk['read_2'] = chunk['read_2'].astype(str)
    chunk['read_3'] = chunk['read_3'].astype(str)
    read2_read3['READV2_CODE'] = read2_read3['READV2_CODE'].astype(str)
    read2_read3['READV3_CODE'] = read2_read3['READV3_CODE'].astype(str)
    ctv_icd_map['read_code'] = ctv_icd_map['read_code'].astype(str)

    read2_part = chunk.loc[~chunk['event_dt'].isna() & ~chunk['read_2'].isna()]
    read3_part = chunk.loc[~chunk['event_dt'].isna() & ~chunk['read_3'].isna()]
    merge_chunk2 = read2_part[["eid","read_2",'event_dt']].merge(read2_read3[['READV2_CODE', 'READV3_CODE']], left_on='read_2', right_on='READV2_CODE')
    merge_chunk2 = merge_chunk2.merge(ctv_icd_map, left_on='READV3_CODE', right_on='read_code')
    merge_chunk3 = read3_part.merge(ctv_icd_map, left_on="read_3", right_on="read_code")
    data_chunks.append(merge_chunk2[['eid', 'icd10_code','event_dt']])
    data_chunks.append(merge_chunk3[['eid', 'icd10_code','event_dt']])
readv3_icd = pd.concat(data_chunks, axis=0)
readv3_icd.drop_duplicates(inplace=True)
print("The length of readv3_icd %d"%(len(readv3_icd)))
df_icd = readv3_icd.groupby('eid')[['icd10_code','event_dt']].agg(list)

def format_yyyymmdd(date):
    """
    Convert a date in DD/MM/YYYY format to YYYY-MM-DD format.
    If the date is already in YYYY-MM-DD format, return it as is.
    If the date is NaN, empty, or not in the expected format, return None.
    """
    
    # Check if the date is NaN or an empty string
    if pd.isnull(date) or date == '':  
        return None

    # If the date is already in YYYY-MM-DD format, return it directly
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date):  # Example: 2024-12-09
        return date

    # If the date is in DD/MM/YYYY format, convert it to YYYY-MM-DD
    if re.match(r'^\d{2}/\d{2}/\d{4}$', date):  # Example: 09/12/2024
        try:
            date_parts = date.split('/')
            return f'{date_parts[2]}-{date_parts[1]}-{date_parts[0]}'  # Convert to YYYY-MM-DD
        except Exception as e:
            print(f"Error processing date: {date}, Error: {e}")  # Print error message for debugging
            return None

tpp_icd_dict = {}
for i in df_icd.index:
    temp={}
    for s,code in enumerate(df_icd['icd10_code'][i]):
        t=format_yyyymmdd(df_icd['event_dt'][i][s])

        if t is None:
            continue
        if code in temp:
            if temp[code] is not None and t < temp[code]:
                temp[code] = t
        else:
            temp[code] = t
    tpp_icd_dict[i]=temp
pickle.dump(tpp_icd_dict, open('../../../results/cache/pricare', 'wb'))

tpp_icd_dict = {}
for i in df_icd.index:
    temp={}
    for s,code in enumerate(df_icd['icd10_code'][i]):
        t=format_yyyymmdd(df_icd['event_dt'][i][s])

        if t is None:
            continue
        if code in temp:
            temp[code].append(t)
        else:
            temp[code] = [t]
        # if code in temp.keys():
        #     temp[code].append(t)
        # else:
        #     temp[code]=[t]
    tpp_icd_dict[i]=temp
pickle.dump(tpp_icd_dict, open('../../../results/cache/pricarer2', 'wb'))