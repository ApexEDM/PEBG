import os 
import pandas as pd
import numpy as np
from scipy import sparse 


def save_dict(dict_name, file_name):
    f = open(file_name, 'w')
    f.write(str(dict_name))
    f.close


def proc_csv():
    file_1 = 'EdNet-Contents/contents/questions.csv'
    df = pd.read_csv(file_1)
    print('original question number: ', len(df))

    # df1 = df[~df['tags'].isin(['-1'])]
    df.drop(df[df['tags']=='-1'].index, inplace=True)
    print('remove non-skill, question number: ', len(df))
    df.to_csv('questions_processed.csv')


def pro_skill_graph():
    df = pd.read_csv('questions_processed.csv')

    pro_id_dict = {}
    pro_ans_dict = {}
    skill_id_dict = {}
    pro_skill_dict = {}
    pro_skill_adj = []
    skill_cnt = 0
    for i, row in df.iterrows():
        # print(i, len(df), row['tags'])
        pro_id_dict[row['question_id']] = i
        pro_ans_dict[row['question_id']] = row['correct_answer']
        tmp_skills = row['tags']
        pro_skill_dict[row['question_id']] = tmp_skills
        for s in tmp_skills.split(';'):
            if s not in skill_id_dict:
                skill_id_dict[s] = skill_cnt
                skill_cnt += 1
            pro_skill_adj.append([i, skill_id_dict[s], 1])

    pro_skill_adj = np.array(pro_skill_adj).astype(np.int32)  
    pro_num = np.max(pro_skill_adj[:, 0]) + 1
    skill_num = np.max(pro_skill_adj[:, 1]) + 1
    print('problem number %d, skill number %d' % (pro_num, skill_num), i)
    # save pro-skill-graph in sparse matrix form
    pro_skill_sparse = sparse.coo_matrix((pro_skill_adj[:, 2].astype(np.float32), (pro_skill_adj[:, 0], pro_skill_adj[:, 1])), shape=(pro_num, skill_num))
    sparse.save_npz('pro_skill_sparse.npz', pro_skill_sparse)

    # take joint skill as a new skill
    skills = df['tags'].unique()
    for s in skills:
        if ';' in s:
            skill_id_dict[s] = skill_cnt
            skill_cnt += 1 

    # save pro-id-dict, skill-id-dict
    save_dict(pro_id_dict, 'pro_id_dict.txt')
    save_dict(pro_ans_dict, 'pro_ans_dict.txt')
    save_dict(pro_skill_dict, 'pro_skill_dict.txt')
    save_dict(skill_id_dict, 'skill_id_dict.txt')


def user_inter(max_stu_num=5000):
    with open('skill_id_dict.txt', 'r') as f:
        skill_id_dict = eval(f.read()) 
    with open('pro_id_dict.txt', 'r') as f:
        pro_id_dict = eval(f.read())
    with open('pro_ans_dict.txt', 'r') as f:
        pro_ans_dict = eval(f.read())
    with open('pro_skill_dict.txt', 'r') as f:
        pro_skill_dict = eval(f.read())

    folder = 'KT1'
    files = os.listdir(folder)
    print(len(files))

    problems = list(pro_id_dict.keys())
    pro_time_dict = {}
    user_inters = []
    cnt_stu_num = 0
    for f in files:
        path = os.path.join(folder, f)
        print(cnt_stu_num, path)

        tmp_inter = pd.read_csv(path, nrows=200)
        tmp_inter = tmp_inter[tmp_inter['question_id'].isin(problems)]
        if len(tmp_inter) < 3:
            continue

        tmp_problems = list(tmp_inter['question_id'])
        tmp_ans_ = list(tmp_inter['user_answer'])
        tmp_time = list(tmp_inter['elapsed_time'])

        tmp_ans = []
        tmp_skills = []
        for i, p in enumerate(tmp_problems):
            a = int(pro_ans_dict[p]==tmp_ans_[i])
            tmp_ans.append(a)
            tmp_skills.append(skill_id_dict[pro_skill_dict[p]])
            if p in pro_time_dict:
                pro_time_dict[p][0] += tmp_time[i] # total time
                pro_time_dict[p][1] += 1   # total number
                pro_time_dict[p][2] += a   # correct number
            else:
                pro_time_dict[p] = [tmp_time[i], 1, a]

        tmp_problems = [pro_id_dict[ele] for ele in tmp_problems]
        user_inters.append([[len(tmp_inter)], tmp_skills, tmp_problems, tmp_ans])

        if cnt_stu_num > max_stu_num:
            break
        cnt_stu_num += 1
    
    print('problem number: ', len(pro_id_dict), len(pro_skill_dict), len(pro_ans_dict), len(pro_time_dict))
    pro_features = np.zeros([len(pro_skill_dict), 2])
    for k, v in pro_time_dict.items():
        pro_features[pro_id_dict[k]] = [v[0]/float(v[1]+1e-4), v[2]/float(v[1]+1e-4)]
    pro_features = np.array(pro_features).astype(np.float32)
    pro_features[:, 0] = (pro_features[:, 0] - np.min(pro_features[:, 0])) / (np.max(pro_features[:, 0])-np.min(pro_features[:, 0]))
    np.savez('pro_feat.npz', pro_feat=pro_features)

    print('user number: ', len(user_inters))
    with open('data.txt', 'w') as f:
        for dd in user_inters:
            for d in dd:
                f.write(str(d)+'\n')


def read_user_sequence(filename='data.txt', max_len=200, min_len=3):
    with open(filename, 'r') as f:
        lines = f.readlines()   
    with open('skill_id_dict.txt', 'r') as f:
        skill_id_dict = eval(f.read()) 
    with open('pro_id_dict.txt', 'r') as f:
        pro_id_dict = eval(f.read())

    y, skill, problem, real_len = [], [], [], []
    skill_num, pro_num = len(skill_id_dict), len(pro_id_dict)
    print("skill number, pro number, ", skill_num, pro_num)

    index = 0
    while index < len(lines):
        num = eval(lines[index])[0]
        tmp_skills = eval(lines[index+1])[:max_len]
        tmp_skills = [ele+1 for ele in tmp_skills]      
        tmp_pro = eval(lines[index+2])[:max_len]
        tmp_pro = [ele+1 for ele in tmp_pro]
        tmp_ans = eval(lines[index+3])[:max_len]

        if num>=min_len:
            tmp_real_len = len(tmp_skills)
            # Completion sequence
            tmp_ans += [-1]*(max_len-tmp_real_len)
            tmp_skills += [0]*(max_len-tmp_real_len)
            tmp_pro += [0]*(max_len-tmp_real_len)

            y.append(tmp_ans)
            skill.append(tmp_skills)
            problem.append(tmp_pro)
            real_len.append(tmp_real_len)

        index += 4
    
    y = np.array(y).astype(np.float32)
    skill = np.array(skill).astype(np.int32)
    problem = np.array(problem).astype(np.int32)
    real_len = np.array(real_len).astype(np.int32)

    print(skill.shape, problem.shape, y.shape, real_len.shape)      
    print(np.max(y), np.min(y))
    print(np.max(real_len), np.min(real_len))  
    print(np.max(skill), np.min(skill))
    print(np.max(problem), np.min(problem))  

    np.savez("ednet.npz", problem=problem, y=y, skill=skill, real_len=real_len, skill_num=skill_num, problem_num=pro_num)


if __name__ == '__main__':
    # proc_csv()
    # pro_skill_graph()
    user_inter()
    read_user_sequence()