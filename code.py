def preprocess_loop_detector_data(file_name):
    df_event_detail = pd.read_csv(file_name, engine='python')

    df_event_detail = df_event_detail.drop(axis=1, columns=[' '])
    dict1 = {}
    for column in df_event_detail.columns:
        dict1[column] = column.strip()
    df_event_detail = df_event_detail.rename(columns=dict1)

    df_event_detail['Timestamp'] = df_event_detail['Timestamp'].apply(lambda x: x.strip())
    df_event_detail['Category'] = df_event_detail['Category'].apply(lambda x: x.strip())
    df_event_detail['Type'] = df_event_detail['Type'].apply(lambda x: x.strip())
    df_event_detail['Timestamp'] = pd.to_datetime(df_event_detail['Timestamp'])
    df_event_detail = df_event_detail.rename(columns={'Event Value':'Event_value'})
    df_event_detail = df_event_detail[(df_event_detail['Timestamp']>=Timestamp('2023-04-19 15:00:00'))&(df_event_detail['Timestamp']<=Timestamp('2023-04-19 17:45:00'))].reset_index(drop=True)
    
    return df_event_detail

def preprocess_microwave_detector_data(file_name):
    df = pd.read_csv(file_name)
    df['measurement_start_timestamp'] = df['measurement_start'].apply(lambda x:str(x.split('T')[0]) + str(' ')+str(x.split('T')[1].split('-')[0]))
    df = df.dropna(subset=['speed', 'volume', 'occupancy'])
    df = df[(df['speed']>=0) & (df['volume']>=0) & (df['occupancy']>=0)]
    df['measurement_start_timestamp'] = pd.to_datetime(df['measurement_start_timestamp'])
    df = df.reset_index(drop=True)
    return df

def aggregate_microwave_detector_data(df,t,lane_number):
    df_average_speed = pd.DataFrame({'Average Speed':df.groupby(pd.Grouper(key='measurement_start_timestamp',freq='{}min'.format(t))).speed.mean()}).reset_index()

    df_max_speed = pd.DataFrame({'Max Speed':df.groupby(pd.Grouper(key='measurement_start_timestamp',freq='{}min'.format(t))).speed.max()}).reset_index()


    df_average_flow = pd.DataFrame({'Average Flow':df.groupby(pd.Grouper(key='measurement_start_timestamp',freq='{}min'.format(t))).volume.sum()}).reset_index()


    df_average_occupancy = pd.DataFrame({'Average Occupancy':df.groupby(pd.Grouper(key='measurement_start_timestamp',freq='{}min'.format(t))).occupancy.mean()}).reset_index()
    df_aggregate = pd.merge(df_average_speed, df_average_flow, on='measurement_start_timestamp',how='left')
    df_aggregate = pd.merge(df_aggregate, df_average_occupancy, on='measurement_start_timestamp',how='left')
    df_aggregate = pd.merge(df_aggregate, df_max_speed, on='measurement_start_timestamp',how='left')

    df_aggregate['Average Flow Rate'] = df_aggregate['Average Flow'].apply(lambda x:(x*60)/t)
    df_aggregate['Congestion measure'] = df_aggregate['Average Speed']/df_aggregate['Max Speed']
    df_aggregate = df_aggregate[(df_aggregate['measurement_start_timestamp']>=Timestamp('2023-04-19 15:00:00'))&(df_aggregate['measurement_start_timestamp']<=Timestamp('2023-04-19 17:45:00'))].reset_index(drop=True)
    column_names = ['Timestamp']
    all_columns = list(df_aggregate.columns)
    for name in range(1,len(all_columns)):
        column_names.append(all_columns[name]+ ' Lane {}'.format(lane_number))
    df_aggregate.columns = column_names
    print(df_aggregate)
    return df_aggregate

def green_time_calculator(df,start,end):
    total_green_time = 0
    start_time = 0
    end_time = 0
    for i in range(len(df)):
        if df['Type'][i] == 'Ramp Meter - Begin Red':
            start_time = df['Timestamp'][i]
        else:
            if start_time != 0:
                end_time = df['Timestamp'][i]
                total_green_time += (start_time - end_time).total_seconds()

    #If last event is Begin Red, add green time from start till first event
    if df['Type'][len(df)-1] == 'Ramp Meter - Begin Red':
        total_green_time += (df['Timestamp'][len(df)-1] - start).total_seconds()

    #If first event is Begin Green, add green time from last event till end
    if df['Type'][0] == 'Ramp Meter - Begin Green':
        total_green_time += (end - df['Timestamp'][0]).total_seconds()

    return(total_green_time)

def aggregate_loop_detector_data(df_event_detail, t):
    #df_gr is Aggregate for Ramp metering rate 
    df_gr = pd.DataFrame({'Aggregate ramp metering rate':df_event_detail.query('Type == "Ramp Meter - Lane 1 Active Rate"').groupby([pd.Grouper(key='Timestamp',freq='{}min'.format(t)),'Type']).Event_value.mean()}).reset_index()
    
    #df_gr2 is Mainline (Lane adjacent to ramp) flow rate
    df_gr2 = pd.DataFrame({'Aggregate Mainline num vehicles':df_event_detail.query('Type == "Detector On" & Event_value==11').groupby([pd.Grouper(key='Timestamp',freq='{}min'.format(t)),'Type']).Event_value.count()}).reset_index()
    df_gr2['Aggregate Mainline flow rate'] = df_gr2['Aggregate Mainline num vehicles'].apply(lambda x: x*(60/t))
    
    #df_gr4 is Ramp Demand flow rate
    df_gr4 = pd.DataFrame({'Aggregate Ramp Demand num vehicles':df_event_detail.query('Type == "Detector On" & Event_value==21').groupby([pd.Grouper(key='Timestamp',freq='{}min'.format(t)),'Type']).Event_value.count()}).reset_index()
    df_gr4['Aggregate Ramp Demand flow rate'] = df_gr4['Aggregate Ramp Demand num vehicles'].apply(lambda x: x*(60/t))
    
    #df_gr5 is Ramp Passage flow rate
    df_gr5 = pd.DataFrame({'Aggregate Ramp Passage num vehicles':df_event_detail.query('Type == "Detector On" & Event_value==23').groupby([pd.Grouper(key='Timestamp',freq='{}min'.format(t)),'Type']).Event_value.count()}).reset_index()
    df_gr5['Aggregate Ramp Passage flow rate'] = df_gr5['Aggregate Ramp Passage num vehicles'].apply(lambda x: x*(60/t))

    df_green_time_estimation = df_event_detail.loc[(df_event_detail['Type']=='Ramp Meter - Begin Green') | (df_event_detail['Type']=='Ramp Meter - Begin Red')].reset_index(drop=True)
    
    #df_gr3 is Green Time
    green_time_array = []
    all_start_timestamps = np.array(df_gr2['Timestamp'])
    for i in range(len(df_gr2)):
        s = df_gr2['Timestamp'][i]
        e = df_gr2['Timestamp'][i] + datetime.timedelta(minutes=t)
        df = df_green_time_estimation.loc[(df_green_time_estimation['Timestamp']>= df_gr2['Timestamp'][i])& (df_green_time_estimation['Timestamp']< df_gr2['Timestamp'][i]+datetime.timedelta(minutes=t))].reset_index(drop=True)
        if len(df)==0:
            green_time_array.append(0)
        else:
            green_time_array.append(green_time_calculator(df,s,e))
            
    df_gr3 = pd.DataFrame()
    df_gr3['Timestamp'] = all_start_timestamps
    df_gr3['Green_time'] = green_time_array
    #print(df_gr)
    #print(df_gr3)
    
    df_aggregate = pd.merge(df_gr, df_gr2[['Timestamp','Aggregate Mainline num vehicles','Aggregate Mainline flow rate']], on='Timestamp',how='left')
    df_aggregate = pd.merge(df_aggregate,df_gr3, on='Timestamp',how='left')
    df_aggregate = pd.merge(df_aggregate,df_gr4[['Timestamp','Aggregate Ramp Demand num vehicles','Aggregate Ramp Demand flow rate']], on='Timestamp',how='left')    
    df_aggregate = pd.merge(df_aggregate,df_gr5[['Timestamp','Aggregate Ramp Passage num vehicles','Aggregate Ramp Passage flow rate']], on='Timestamp',how='left') 
    return df_aggregate

def aggregate_all_lanes_microwave(list_of_files,num_lanes,t):
    df_lane_1 = preprocess_microwave_detector_data(list_of_files[0])
    df_lane_1 = aggregate_microwave_detector_data(df_lane_1,t,1)
    
    df_lane_2 = preprocess_microwave_detector_data(list_of_files[1])
    df_lane_2 = aggregate_microwave_detector_data(df_lane_2,t,2)
    
    df_aggregate = pd.merge(df_lane_1,df_lane_2,on='Timestamp')
    
    for i in range(2, num_lanes):
        df = preprocess_microwave_detector_data(list_of_files[i])
        df = aggregate_microwave_detector_data(df,t,i+1)
        df_aggregate = pd.merge(df_aggregate,df,on='Timestamp',how='left')
    return df_aggregate

def preprocess_DeepSORT_data(file_name):
    df = pd.read_csv(file_name)
    df['time1'] = pd.to_datetime(df['time1'])
    df['time2'] = pd.to_datetime(df['time2'])
    return df

def aggregate_DeepSORT_data(df, t, lane_number):
    df_lane = df[df['lane']=='lane{}'.format(lane_number)].reset_index(drop=True)
    df_average_speed = pd.DataFrame({'Average Speed':df_lane.groupby(pd.Grouper(key='time2',freq='{}min'.format(t))).speed.mean()}).reset_index()
    df_max_speed = pd.DataFrame({'Max Speed':df_lane.groupby(pd.Grouper(key='time2',freq='{}min'.format(t))).speed.max()}).reset_index()
    
    df_average_flow = pd.DataFrame({'Number of Vehicles':df_lane.groupby(pd.Grouper(key='time2',freq='{}min'.format(t))).track_id.count()}).reset_index()
    df_average_flow['Average Flow Rate'] = df_average_flow['Number of Vehicles'].apply(lambda x: x*(60/t))
    df_aggregate = pd.merge(df_average_speed, df_average_flow,on='time2',how='left')
    df_aggregate = pd.merge(df_aggregate, df_max_speed, on='time2',how='left')
    df_aggregate['Average Vehicle Density'] = df_aggregate['Average Flow Rate']/df_aggregate['Average Speed']
    df_aggregate['Congestion measure'] = df_aggregate['Average Speed']/df_aggregate['Max Speed']
    df_aggregate = df_aggregate[(df_aggregate['time2']>=Timestamp('2023-04-19 15:00:00'))&(df_aggregate['time2']<=Timestamp('2023-04-19 17:45:00'))].reset_index(drop=True)
    column_names = ['Timestamp']
    all_columns = list(df_aggregate.columns)
    for name in range(1,len(all_columns)):
        column_names.append(all_columns[name]+ ' Lane {}'.format(lane_number)+' DeepSORT')
    df_aggregate.columns = column_names
    return df_aggregate

def aggregate_all_lanes_DeepSORT(file_name, num_lanes, t):
    df_lane1 = preprocess_DeepSORT_data(file_name)
    df_lane1 = aggregate_DeepSORT_data(df_lane1, t, 1)
    df_lane2 = preprocess_DeepSORT_data(file_name)
    df_lane2 = aggregate_DeepSORT_data(df_lane2, t, 2)
    df_aggregate = pd.merge(df_lane1,df_lane2,on='Timestamp')
    
    for i in range(2, num_lanes):
        df = preprocess_DeepSORT_data(file_name)
        df = aggregate_DeepSORT_data(df,t,i+1)
        df_aggregate = pd.merge(df_aggregate,df,on='Timestamp',how='left')
    return df_aggregate
    
def main(num_lanes, t):
    df_loop = preprocess_loop_detector_data('/Users/sainikhil/Downloads/RADISH/2023-04-19 Cypress 101 data/101_Cypress_EventReport_1858.csv')
    df_loop_aggregate = aggregate_loop_detector_data(df_loop,t)
    print(df_loop_aggregate)
    list_of_files = ['/Users/sainikhil/Downloads/RADISH/2023-04-19 Cypress 101 data/MVDS raw data/Lane_Readings_15446-1-45827.csv',
                    '/Users/sainikhil/Downloads/RADISH/2023-04-19 Cypress 101 data/MVDS raw data/Lane_Readings_15446-2-45828.csv',
                    '/Users/sainikhil/Downloads/RADISH/2023-04-19 Cypress 101 data/MVDS raw data/Lane_Readings_15446-3-45829.csv']
    df_microwave_aggregate = aggregate_all_lanes_microwave(list_of_files,num_lanes,t)
    print(df_microwave_aggregate)
    df_deepsort_aggregate = aggregate_all_lanes_DeepSORT('DeepSORT_tracking_Results.csv',num_lanes,t)
    df = pd.merge(df_loop_aggregate,df_microwave_aggregate,on='Timestamp',how='left')
    df = pd.merge(df,df_deepsort_aggregate,on='Timestamp',how='left')
    df.to_csv('101 Cypress Results Data Aggregated over {} minutes interval.csv'.format(t),index=False)