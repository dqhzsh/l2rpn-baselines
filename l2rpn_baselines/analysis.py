import argparse
import os
import pandas
import csv
import bz2
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Import necessary modules
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.PlotGrid import PlotMatplot

#help(grid2op.PlotGrid.PlotMatplot)



# Create the environment
backend = LightSimBackend()
env = grid2op.make("l2rpn_neurips_2020_track2_small", backend=backend)

print(dir(env))
if hasattr(env, '_helper_action_player'):
    print(f"{env} 对象具有 _helper_action_player 属性")
else:
    print(f"{env} 对象没有 _helper_action_player 属性")


print(dir(env.action_space))

env_1 =grid2op.make("l2rpn_case14_sandbox", backend=backend)
print(dir(env_1))
if hasattr(env_1, '_init_grid_path'):
    print(f"{env_1} 对象具有 _init_grid_path 属性")
else:
    print(f"{env_1} 对象没有 _init_grid_path 属性")



import random
def random_task(env, N_task):
    tasks = []
    # Loop for N_task times
    for _ in range(N_task):
        # Randomly select an environment
        mix_names = list(env.keys())
        random_env_name = random.choice(mix_names)
        random_env = env[random_env_name]

        # Get all chronic names
        all_chronics_paths = random_env.chronics_handler.subpaths
        all_chronics_names = [path.split("\\")[-1] for path in all_chronics_paths]

        # Randomly select a chronic
        random_chronic_name = random.choice(all_chronics_names)

        # Set the environment to the selected chronic
        random_env.set_id(random_chronic_name)

        # Reset the environment
        random_env.reset()

        tasks.append(random_env)

    print(tasks)
    print(tasks[2].chronics_handler.get_name())

random_task(env,10)


# list all available mixes:
mixes_names = list(env.keys())
print(list(env.keys()))
#env.set_id("Scenario_august_44")

# and now supposes we want to study only the first one
mix = env[mixes_names[1]]
print(mix.env_name)



mix.set_id("Scenario_april_34")
mix.reset()
name = mix.chronics_handler.get_name()
print("当前 ID:", mix.chronics_handler.get_name())



# 创建chronics处理器
chronics_handler = mix.chronics_handler

total_episode = len(chronics_handler.subpaths)
print(total_episode)


# 获取所有chronic的路径
all_chronics_paths = chronics_handler.subpaths

# 获取所有chronic的名称
all_chronics_names = [path.split("\\")[-1] for path in all_chronics_paths]

# 打印所有chronic的名称
print("所有chronic的名称：", all_chronics_names)

for id_ in all_chronics_names:
    mix.set_id(id_)  # tell the environment you simply want to use the chronics with ID 0
    mix.reset()
    print("当前 ID:", mix.chronics_handler.get_name())



# # Set the episode ID, reset the environment, and get the observation
# #env.set_id(0)
# obs = env.reset()
#
# # Create a PlotMatplot instance
# plot_helper = PlotMatplot(env.observation_space)
#
# # Plot the initial observation
# fig = plot_helper.plot_obs(obs, storage_info=None, gen_info=None, load_info=None, line_info=None)
#
#
# fig.show()
# fig.savefig(f"Grid {name}.pdf")






# def decompress_and_read_bz2_file(compressed_file_path):
#     with bz2.open(compressed_file_path, 'rt') as decompressed_file:
#         file_content = decompressed_file.read()
#     return file_content
#
#
# def list_files_root(directory):
#     files = []
#     with os.scandir(directory) as entries:
#         for entry in entries:
#             if entry.is_file():
#                 files.append(entry.name)
#     return files
#
#
# def list_subdirectories(directory):
#     subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
#     return subdirectories
#
#
# def getPType(col_sp, prods_charac_data):
#     # 检查条件是否至少有一个匹配项
#     if not prods_charac_data.loc[prods_charac_data['name'] == col_sp].empty:
#         # 如果至少有一个匹配项，获取第一个匹配项的行号
#         row = prods_charac_data.loc[prods_charac_data['name'] == col_sp].index[0]
#         return prods_charac_data.loc[row, 'type']
#     else:
#         print(f"No match found for {col_sp} in the name column.")
#         return -1
#
# def sort_months(month_list):
#     # 定义月份的顺序映射
#     month_order_mapping = {
#         'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
#         'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
#     }
#
#     # 使用 lambda 函数从月份字符串中提取月份
#     extract_month = lambda month: month.split('_')[1]
#
#     # 根据月份顺序和映射的数字进行排序
#     sorted_months = sorted(month_list, key=lambda x: (month_order_mapping[extract_month(x)], x))
#
#     return sorted_months
#
# if __name__ == "__main__":
#     path = "C:/Users/dqh/Desktop/l2rpn_neurips_2020_track2_small/l2rpn_neurips_2020_track2_small"
#     # 存储所有 dir 的数据
#     all_data = []
#     all_monthly_data=[]
#     for dir in list_subdirectories(path):
#         prods_charac_path = path + "/" + dir + "/" + "prods_charac.csv"
#         prods_charac_data = pandas.read_csv(prods_charac_path, encoding='gbk')
#         sum_wind_year = 0
#         sum_solar_year = 0
#         sum_all_year = 0
#         all_clean_percent_min = 1
#         all_clean_percent_max = 0
#         all_clean_percent_list = []  # 用于存储每个月的 all_clean_percent 值
#         solar_percent_list = []
#         wind_percent_list = []
#         month_list = sort_months(list_subdirectories(path + "/" + dir + "/" + "chronics"))
#
#         for dir1 in month_list:
#             sum_wind_month = 0
#             sum_solar_month = 0
#             sum_all_month = 0
#
#             filepath = path + "/" + dir + "/" + "chronics" + "/" + dir1 + "/" + "prod_p.csv.bz2"
#             data = pandas.read_csv(filepath)
#             key = data.columns[0]
#             sp_keys = key.split(";")
#             wind_indexs = []
#             solar_indexs = []
#             for index, col_sp in enumerate(sp_keys):
#                 getType = getPType(col_sp, prods_charac_data)
#                 if getType == 'wind':
#                     wind_indexs.append(index)
#                 elif getType == 'solar':
#                     solar_indexs.append(index)
#             first_column_values = data.iloc[:, 0].tolist()
#             for first_column_value in first_column_values:
#                 sp_values = first_column_value.split(";")
#                 for index, sp_value in enumerate(sp_values):
#                     if index in wind_indexs:
#                         sum_wind_month += float(sp_value)
#                     elif index in solar_indexs:
#                         sum_solar_month += float(sp_value)
#                     sum_all_month += float(sp_value)
#             sum_wind_year += sum_wind_month
#             sum_solar_year += sum_solar_month
#             sum_all_year += sum_all_month
#             print("root=" + dir + ",month=" + dir1 + ",sum_all_month="
#                   + str(sum_all_month) + ",sum_wind_month=" + str(sum_wind_month)
#                   + ",sum_solar_month=" + str(sum_solar_month)
#                   + ",wind_percent=" + str(sum_wind_month / sum_all_month)
#                   + ",solar_percent=" + str(sum_solar_month / sum_all_month)
#                   + ",all_clean_percent=" + str((sum_solar_month + sum_wind_month) / sum_all_month))
#             all_clean_percent = (sum_solar_month + sum_wind_month) / sum_all_month
#             all_clean_percent_list.append(all_clean_percent)
#             solar_percent_list.append(sum_solar_month / sum_all_month)
#             wind_percent_list.append(sum_wind_month / sum_all_month)
#             if all_clean_percent < all_clean_percent_min:
#                 all_clean_percent_min = all_clean_percent
#             if all_clean_percent > all_clean_percent_max:
#                 all_clean_percent_max = all_clean_percent
#
#
#
#
#
#
#
#         # 创建一个新的图形窗口
#         plt.figure()
#         # 绘制 clean_percent_list 的线，使用蓝色
#         plt.plot(all_clean_percent_list, label=f'{dir} - all_clean_percent', color='blue')
#
#         # 绘制 sum_solar_month 的线，使用红色
#         plt.plot(solar_percent_list, label=f'{dir} - solar_percent', color='red')
#
#         # 绘制 sum_wind_month 的线，使用绿色
#         plt.plot(wind_percent_list, label=f'{dir} - wind_percent', color='green')
#
#         plt.xlabel('Month')
#         plt.ylabel('clean Percent')
#         # 设置纵坐标标签格式为百分比
#         def to_percent(y, position):
#             return f'{y * 100:.0f}%'
#         plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#         plt.ylim(0, 1)
#         plt.title('clean Percent Over Time')
#         plt.text(1, 0.7, f'all_all_clean_percent_average:{((sum_solar_year + sum_wind_year) / sum_all_year)*100:.2f}%, range from{all_clean_percent_min*100:.2f}%~{all_clean_percent_max*100:.2f}%', fontsize=8, color='blue')
#         plt.legend()
#         plt.show()
#         print("------------------------all---------------------------")
#         print("dir=" + dir + ",sum_wind_year=" + str(sum_wind_year) + "sum_solar_year" + str(
#             sum_wind_year) + "sum_all_year" + str(sum_all_year)
#               + ",all_wind_percent=" + str(sum_wind_year / sum_all_year)
#               + ",all_solar_percent=" + str(sum_solar_year / sum_all_year)
#               + ",all_all_clean_percent_average=" + str((sum_solar_year + sum_wind_year) / sum_all_year))
#         print("------------------------all---------------------------")
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#         #seaborn画图
#
#         # Assuming all_clean_percent_list is the list you provided
#
#         # Split the list into groups of 10
#         monthly_data = [all_clean_percent_list[i:i + 10] for i in range(0, len(all_clean_percent_list), 10)]
#
#         # Create a DataFrame from the grouped data
#         df = pandas.DataFrame(monthly_data, columns=[f'DataPoint_{i+1}' for i in range(10)])
#
#         # Add a column for the month
#         months = [f'Month_{i + 1}' for i in range(len(monthly_data))]
#         df['Month'] = months
#
#         # Melt the DataFrame to reshape it for Seaborn
#         df_melted = pandas.melt(df, id_vars='Month', var_name='Day', value_name='All Clean Percent')
#
#         # Seaborn line plot with shadows
#         plt.figure(figsize=(12, 6))
#         sns.lineplot(x='Month', y='All Clean Percent', data=df_melted, ci='sd', err_style='band', palette='viridis')
#         def to_percent(y, position):
#             return f'{y * 100:.0f}%'
#         plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#         plt.ylim(0, 1)
#         plt.xlabel('Month')
#         plt.ylabel('All Clean Percent')
#         plt.title(f'{dir} - all_clean_percent')
#
#         plt.show()
#
#
#
#         # seaborn绘制所有dir一起的图
#
#         # 存储当前 dir 的数据
#         dir_monthly_data = {
#             'dir': dir,
#             'all_monthly_data': monthly_data,
#         }
#
#         # 将当前 dir 的数据添加到 all_data 中
#         all_monthly_data.append(dir_monthly_data)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#         #绘制所有dir一起的图
#         plt.figure()
#
#         # 存储当前 dir 的数据
#         dir_data = {
#             'dir': dir,
#             'all_clean_percent_list': all_clean_percent_list,
#             'solar_percent_list': solar_percent_list,
#             'wind_percent_list': wind_percent_list,
#             'sum_solar_year': sum_solar_year,
#             'sum_wind_year': sum_wind_year,
#             'sum_all_year': sum_all_year,
#             'all_clean_percent_min': all_clean_percent_min,
#             'all_clean_percent_max': all_clean_percent_max
#         }
#
#         # 将当前 dir 的数据添加到 all_data 中
#         all_data.append(dir_data)
#
#
#
#
#     # 绘制所有 dir 的图
#     for dir_data in all_data:
#         plt.plot(dir_data['all_clean_percent_list'], label=f'{dir_data["dir"]} - all_clean_percent')
#     plt.xlabel('Month')
#     plt.ylabel('clean Percent')
#
#
#     # 设置纵坐标标签格式为百分比
#     def to_percent(y, position):
#         return f'{y * 100:.0f}%'
#
#
#     plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#     plt.ylim(0, 1)
#     plt.title('clean Percent Over Time')
#
#     plt.legend()
#     plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#     # seaborn绘制所有dir一起的图
#     dfs = []
#
#     for dir_monthly_data in all_monthly_data:
#         df = pandas.DataFrame(dir_monthly_data['all_monthly_data'], columns=[f'DataPoint_{i + 1}' for i in range(10)])
#         df['Month'] = [f'Month_{i + 1}' for i in range(len(dir_monthly_data['all_monthly_data']))]
#
#         # Melt the DataFrame to reshape it for Seaborn
#         df_melted = pandas.melt(df, id_vars='Month', var_name='Day', value_name='All Clean Percent')
#
#         # Add a column for the dir
#         df_melted['Dir'] = dir_monthly_data['dir']
#
#         # Append the dataframe to the list
#         dfs.append(df_melted)
#
#     # Concatenate all the dataframes into a single dataframe
#     final_df = pandas.concat(dfs)
#
#     # Seaborn line plot with shadows
#     plt.figure(figsize=(12, 6))
#     sns.lineplot(x='Month', y='All Clean Percent', hue='Dir', data=final_df, ci='sd', err_style='band',
#                  palette='viridis')
#
#
#     def to_percent(y, position):
#         return f'{y * 100:.0f}%'
#
#
#     plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
#     plt.ylim(0, 1)
#     plt.xlabel('Month')
#     plt.ylabel('All Clean Percent')
#     plt.title('All Dirs - all_clean_percent')
#     plt.show()
