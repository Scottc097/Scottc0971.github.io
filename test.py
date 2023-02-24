# checking if files all 12 months or if I will have to concat all files together
import calendar

import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from geopy.distance import geodesic

#
# # concat
# # Define a list of file names
# files = ["C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202004-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202005-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202006-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202007-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202008-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202009-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202010-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202011-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202012-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202101-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202102-divvy-tripdata.csv",
#          "C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/202103-divvy-tripdata.csv"]
#
# # Load the first file into a DataFrame
# df_concat = pd.read_csv(files[0])
#
# # Loop through the rest of the files and concatenate them to the first DataFrame
# for i in range(1, len(files)):
#     df = pd.read_csv(files[i])
#     df_concat = pd.concat([df_concat, df], axis=0)
#
# # Save the concatenated DataFrame to a new CSV file
# df_concat.to_csv("concatenated_file.csv", index=False)


# checking if files all 12 months or if I will have to concat all files together
# Read the CSV file into a pandas DataFrame
df = pd.read_csv("C:/Users/Scott/Documents/Google Certificate/Case 1 Study/Updated/concatenated_file.csv", dtype={"column_5": "Int64", "column_7": str})

print(df.head())  # to check the first few rows of the dataset
print(df.describe())  # to get some statistics about the data
print(df.info())  # to get information about the data types and missing values

# replace missing values with "N/A"
df.fillna("N/A", inplace=True)

# replace blank entries with NaN values
df = df.replace('', pd.NA)

# check for duplicates and remove them
df.drop_duplicates(inplace=True)

# rename columns
df.rename(columns={'rideable_type': 'vehicle_type', 'member_casual': 'user_type'}, inplace=True)


##type
#turning columns into correct type
# convert the "started_at" and "ended_at" columns to datetime
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])
df['start_lat'] = df['start_lat'].astype(float)
df['start_lng'] = df['start_lng'].astype(float)
df['end_lat'] = df['end_lat'].replace('N/A', np.nan).astype(float)
df['end_lng'] = df['end_lng'].replace('N/A', np.nan).astype(float)

##Variables
# calculate the ride length as the difference between "ended_at" and "started_at" columns
df['ride_length'] = df['ended_at'] - df['started_at']
# Convert ride length to seconds
df['ride_length_seconds'] = df['ride_length'].dt.total_seconds()
# Convert ride_length to minutes
df['ride_length_minutes'] = df['ride_length'].dt.total_seconds() / 60
# extract hour of day from started_at column
df['hour_of_day'] = pd.to_datetime(df['started_at']).dt.hour
# Extract the month from a date column
df['month'] = pd.to_datetime(df['started_at']).dt.month
# calculate the day of the week that each ride started using the "weekday" command
df['day_of_week'] = df['started_at'].dt.weekday
# format the "day_of_week" column as a number with no decimals
df['day_of_week'] = df['day_of_week'].astype(int)
# Extract the year from the 'started_at' column
df['year'] = pd.DatetimeIndex(df['started_at']).year

##Dictionaries
# create a dictionary to map day of week numbers to day names
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
# create a dictionary to map hour of day numbers to hour values
hour_values = {0: '12 a.m.', 1: '1 a.m.', 2: '2 a.m.', 3: '3 a.m.', 4: '4 a.m.', 5: '5 a.m.', 6: '6 a.m.', 7: '7 a.m.', 8: '8 a.m.', 9: '9 a.m.', 10: '10 a.m.', 11: '11 a.m.', 12: '12 p.m.', 13: '1 p.m.', 14: '2 p.m.', 15: '3 p.m.', 16: '4 p.m.', 17: '5 p.m.', 18: '6 p.m.', 19: '7 p.m.', 20: '8 p.m.', 21: '9 p.m.', 22: '10 p.m.', 23: '11 p.m.'}
colors = ['blue', 'orange']  # Define colors for each user type


# Count the unique values in the month column
unique_months = df['month'].nunique()

print("Number of unique months:", unique_months)

# count the number of rides with a ride length over 1 hour
num_rides_over_1h = (df['ride_length'] > pd.Timedelta(hours=1)).sum()
# Count the number of rides with a ride length over 500 hours
num_rides_over_500h = (df['ride_length'] > pd.Timedelta(hours=500)).sum()
# Count the number of rides with a ride length over 500 hours
num_rides_under_1h = (df['ride_length'] < pd.Timedelta(hours=1)).sum()
print(f"Number of rides with a ride length over 500 hours: {num_rides_over_500h}")
print(f"Number of rides with a ride length over 1 hours: {num_rides_over_1h}")
print(f"Number of rides with a ride length under 1 hours: {num_rides_under_1h}")

# filter out rides with length greater than 1 hour
df = df[(pd.Timedelta(hours=1) >= df['ride_length']) & (df['ride_length'] >= pd.Timedelta(seconds=0))]


# # Save the concatenated DataFrame to a new CSV file
# df.to_csv('Modified_Bike_Data_V01.csv', index=False)


# Unique values for categorical columns
print(df['vehicle_type'].unique())
print(df['user_type'].unique())

# Frequency counts for categorical columns
print(df['vehicle_type'].value_counts())
print(df['user_type'].value_counts())


# calculate the mean of ride_length
mean_ride_length = df['ride_length'].mean()

print(f"Mean ride length time: {mean_ride_length}")

# calculate the max of ride_length
max_ride_length = df['ride_length'].max()
print(f"Max ride length: {max_ride_length}")

# calculate the mode of day_of_week
mode_day_of_week = df['day_of_week'].mode()[0]
print(f"Mode day of week: {mode_day_of_week}")


# Group the data by user_type and calculate the mean of ride_length for each group
ride_length_by_user_type = df.groupby('user_type')['ride_length'].mean()

# Print the results
print("Average ride length for members:", ride_length_by_user_type['member'])
print("Average ride length for casual riders:", ride_length_by_user_type['casual'])

# calculate the average ride length for users by day of week
avg_ride_length_by_day = df.groupby('day_of_week')['ride_length'].mean()

# print the result
print(f"Average ride length by day of week:\n{avg_ride_length_by_day}")

##Bar Chart displaying distribution of vehicle types and user types
# Group the data by vehicle type and user type, and count the number of rides in each group
ride_counts1 = df.groupby(['vehicle_type', 'user_type'])['ride_id'].count()


# Create a bar chart and use the colormap to set the colors of the bars
ax = ride_counts1.unstack().plot(kind='bar', colormap='Paired')

# Set the title and labels
plt.title('Rides by Vehicle Type and User Type')
plt.xlabel('Vehicle and User Type')
plt.ylabel('Number of Rides')

# Set the legend
handles, labels = ax.get_legend_handles_labels()
labels = [f'{label} ({ride_counts1.loc[:, label].sum()})' for label in labels]
ax.legend(handles, labels, title='User Type')

# Show the plot
plt.show()

##Pie chart with distribution of vehicle and user types
# Create a pie chart
ride_counts1.plot(kind='pie', autopct='%1.1f%%')
plt.title('Rides by Vehicle Type and User Type')
plt.axis('equal')
plt.legend(title='Vehicle and User Type', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

##Histogram: To show the distribution of ride lengths for members and casual riders and overlaying
# Filter the data by user type
members_data = df[df['user_type'] == 'member']
casual_data = df[df['user_type'] == 'casual']


# Create separate histograms for members and casual riders
plt.hist(df.loc[df['user_type'] == 'member', 'ride_length_seconds'], bins=20, alpha=0.5, label='Member')
plt.hist(df.loc[df['user_type'] == 'casual', 'ride_length_seconds'], bins=20, alpha=0.5, label='Casual')

# Add labels and legend
plt.xlabel('Ride Length (Seconds)')
plt.ylabel('Frequency')
plt.legend()

# Display the histogram
plt.show()



# Create a box plot of ride lengths
plt.boxplot(df['ride_length_minutes'])
plt.ylabel('Ride Length (minutes)')
plt.title('Box Plot of Ride Lengths')
plt.show()


##Line plot of showing mean ride length on days of week
# Group the data by weekday and calculate the mean ride length for each weekday
rides_by_weekday = df.groupby('day_of_week')['ride_length'].mean()

# Create a line plot
rides_by_weekday.plot(kind='line')

# Set the x-axis labels
plt.xticks(range(0, 7), day_names)

# Set the title, x-axis label, and y-axis label
plt.title('Average Ride Length by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Ride Length (seconds)')

# Show the plot
plt.show()


#  # Save the concatenated DataFrame to a new CSV file
# df.to_csv('Modified_Bike_Data_V02.csv', index=False)

# create pivot table
pivot_table = pd.pivot_table(df, values='ride_length_minutes', index='day_of_week', columns='hour_of_day', aggfunc='mean')

# create heatmap
plt.figure(figsize=(12,6))
sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.1f', linewidths=.5)
plt.title('Average Ride Length (minutes) by Day of Week and Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')

# add a color-coded legend showing the mapping of day of week numbers to day names and hour values
plt.xticks(np.arange(0.5, 24.5, 1), [hour_values[i] for i in range(24)])
plt.yticks(np.arange(0.5, 7.5, 1), [day_names[i] for i in range(7)])
plt.show()




# drop rows where any of the start or end variables have NaN values
df = df.dropna(subset=['start_lat', 'start_lng', 'end_lat', 'end_lng'])

# calculate ride distance and duration
df['ride_distance1'] = df.apply(lambda row: geodesic((row['start_lat'], row['start_lng']), (row['end_lat'], row['end_lng'])).miles, axis=1)
df['ride_duration_minutes'] = df['ride_length'] / pd.Timedelta(minutes=1)
# print first 10 rows of the dataframe
print(df.head(10))



# Segment the data by user type
by_user = df.groupby("user_type")

#ride distance mean by user
result = by_user.agg({"ride_distance1": "mean"})

# Display the result
print(result)




def get_ride_frequency(ride_id):
    try:
        return int(ride_id[13])
    except ValueError:
        return None

df['ride_frequency'] = df['ride_id'].apply(get_ride_frequency)


# Create a new DataFrame with the count of rides per user
ride_counts = df.groupby('user_type')['ride_id'].count().reset_index(name='ride_count')

# Plot the ride counts by user type
sns.barplot(x='user_type', y='ride_count', data=ride_counts)
plt.title('Ride Counts by User Type')
plt.xlabel('User Type')
plt.ylabel('Ride Count')
plt.show()



# Group data by user_id and count the number of rides
ride_counts = df.groupby("user_type")["ride_id"].count()

# Print summary statistics of the ride counts
print(ride_counts.describe())


# Segment the data by month and day of the week
by_month_weekday = df.groupby(["month", "day_of_week"])

# Calculate the total number of rides for each combination
result = by_month_weekday.agg({"ride_id": "count"})

# Reset the index and create a new column with the formatted label
result = result.reset_index()
result["label"] = result["month"].apply(lambda x: calendar.month_abbr[x]) + " " + result["day_of_week"].apply(lambda x: calendar.day_abbr[x])


# Display the result
print(result)




# Step 3: Segment the data
df['ride_length_minutes'] = df['ride_length_seconds'] / 60
df['ride_distance_km'] = df['ride_distance1'] / 1000
df['user_type_category'] = df['user_type'].apply(lambda x: 'Member' if x == 'member' else 'Casual')
df['hour_of_day_category'] = pd.cut(df['hour_of_day'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

# Step 4: Analyze the segmented data
ride_frequency = df['user_type_category'].value_counts()


ride_frequency.plot(kind='bar', color=colors)

plt.title('Ride Frequency by User Type')
plt.xlabel('User Type')
plt.ylabel('Frequency')
plt.show()

ride_distance = df.groupby('user_type_category')['ride_distance_km'].mean()
ride_distance.plot(kind='bar')
plt.title('Average Ride Distance by User Type')
plt.xlabel('User Type')
plt.ylabel('Distance (km)')
plt.show()


# Define the labels for each hour of day
hour_labels = ['12 AM - 2 AM', '2 AM - 4 AM', '4 AM - 6 AM', '6 AM - 8 AM', '8 AM - 10 AM', '10 AM - 12 PM',
               '12 PM - 2 PM', '2 PM - 4 PM', '4 PM - 6 PM', '6 PM - 8 PM', '8 PM - 10 PM', '10 PM - 12 AM']



#new
# convert ride_length column to timedelta dtype
df['ride_length'] = pd.to_timedelta(df['ride_length'])

# compute mean ride length by user type
ride_length = df.groupby('user_type')['ride_length'].mean()

# convert back to float for plotting
ride_length = ride_length.astype('timedelta64[m]').astype('float64')

# plot the bar chart
ride_length.plot(kind='bar', title='Ride Length by User Type')
plt.xlabel('User Type')
plt.ylabel('Average Ride Length (minutes)')
plt.show()



df['start_hour'] = pd.to_datetime(df['started_at']).dt.hour
ride_time = df.groupby(['user_type', 'start_hour'])['ride_id'].count()

ax = ride_time.unstack('user_type').plot(title='Ride Time of Day by User Type')
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Number of Rides')
ax.legend(title='User Type')
plt.show()




# create a new column that concatenates the start and end station names
df = df.drop(index=df[df['start_station_name'] == 'N/A'].index)
df = df.drop(index=df[df['end_station_name'] == 'N/A'].index)
df['route'] = df['start_station_name'] + ' -> ' + df['end_station_name']
# group the data by user_type and route, and count the number of rides for each group
route_count = df.groupby(['user_type', 'route'])['ride_id'].count()

# sort the results in descending order and select the top 10 routes for each user type
top_routes = route_count.groupby('user_type').apply(lambda x: x.sort_values(ascending=False).head(10))

# print the results
print('Top 10 Routes by User Type:')
print(top_routes)



# Group the data by year and vehicle type and count the ride_id
df_grouped = df.groupby(['year', 'vehicle_type']).count()['ride_id'].reset_index()

# Pivot the data to make vehicle types as columns
df_pivoted = df_grouped.pivot(index='year', columns='vehicle_type', values='ride_id')

# Plot the line graph
df_pivoted.plot.line()
plt.xlabel('Year')
plt.ylabel('Number of Rides')
plt.title('Growth or Decrease of Each Vehicle Type Over the Years')
plt.show()
print(df_pivoted)
print(df_grouped)


# display the chart
# extract the hour component using the dt.hour attribute
df['hour1'] = df['started_at'].dt.hour

# group the DataFrame by the hour and count the number of entries for each hour
hour_counts = df.groupby('hour1').count()

# create a bar chart of the results
hour_counts.plot(kind='bar', legend=None)
plt.xlabel('Hour')
plt.ylabel('Count')
plt.title('Number of Entries by Hour')
plt.show()

# print the DataFrame to see the results
print(hour_counts)


# create a new column that concatenates the start and end station names
df = df.drop(index=df[df['start_station_name'] == 'N/A'].index)
df = df.drop(index=df[df['end_station_name'] == 'N/A'].index)
df['route'] = df['start_station_name'] + ' -> ' + df['end_station_name']



# create a new column that concatenates the start and end station names
df = df.drop(index=df[df['start_station_name'] == 'N/A'].index)
df = df.drop(index=df[df['end_station_name'] == 'N/A'].index)
df['route'] = df['start_station_name'] + ' -> ' + df['end_station_name']
# group the data by user_type and route, and count the number of rides for each group
route_count = df.groupby(['user_type', 'route'])['ride_id'].count()

# sort the results in descending order and select the top 10 routes for each user type
top_routes = route_count.groupby('user_type').apply(lambda x: x.sort_values(ascending=False).head(10))

# print the results
print('Top 10 Routes by User Type:')
print(top_routes)

# select only the desired columns
df = df[['ride_id', 'vehicle_type', 'start_lat', 'start_lng', 'end_lat', 'end_lng', 'user_type','start_station_name','end_station_name']]

# save the filtered dataset to a new CSV file
df.to_csv('Bike_data_Tableau_V02.csv', index=False)
