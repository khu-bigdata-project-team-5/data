import pandas as pd
import glob
import os
import re
import ast

df = pd.read_csv("udemy_en2.csv")
df.rename(columns={'keywords': 'keywords_array'}, inplace=True)

pattern = r'[0-9&\-\(\)\u0600-\u06FF]'

def clean_keywords_array(keywords_str):
    keywords_list = ast.literal_eval(keywords_str)
    cleaned_list = [re.sub(pattern, '', keyword).strip() for keyword in keywords_list]
    cleaned_list = [keyword for keyword in cleaned_list if keyword]
    seen = set()
    unique_list = []
    for keyword in cleaned_list:
        if keyword not in seen:
            unique_list.append(keyword)
            seen.add(keyword)
    return unique_list

df['keywords_array'] = df['keywords_array'].apply(clean_keywords_array)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

development_keywords = set([
    'python', 'java', 'c', 'c++', 'c#', 'javascript', 'typescript', 'ruby', 'php', 'swift', 'kotlin',
    'html', 'css', 'sass', 'less', 'jquery', 'ajax', 'xml', 'json',
    'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'sqlite', 'redis', 'cassandra',
    'data', 'analysis', 'analytics', 'datascience', 'machinelearning', 'deeplearning', 'neuralnetworks',
    'ai', 'artificialintelligence', 'nlp', 'naturallanguageprocessing',
    'cloud', 'aws', 'azure', 'googlecloud', 'gcp', 'ibmcloud', 'oraclecloud',
    'devops', 'docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'puppet', 'chef',
    'git', 'github', 'gitlab', 'bitbucket', 'svn',
    'agile', 'scrum', 'kanban', 'lean', 'xp', 'extremeprogramming',
    'mobile', 'android', 'ios', 'reactnative', 'flutter', 'xamarin',
    'web', 'react', 'angular', 'vue', 'svelte', 'nextjs', 'nuxtjs', 'gatsby', 'bootstrap', 'foundation', 'materialize',
    'backend', 'frontend', 'fullstack', 'api', 'rest', 'graphql', 'grpc',
    'nodejs', 'express', 'django', 'flask', 'spring', 'laravel', 'rails', 'sinatra',
    'testing', 'unittesting', 'integrationtesting', 'e2etesting', 'selenium', 'junit', 'pytest', 'mocha', 'chai',
    'security', 'cybersecurity', 'encryption', 'decryption', 'ssl', 'tls', 'firewall',
    'networking', 'tcp', 'udp', 'ip', 'dns', 'http', 'https',
    'containers', 'virtualization', 'vagrant', 'vmware', 'virtualbox',
    'designpatterns', 'oop', 'functionalprogramming', 'reactiveprogramming',
    'bigdata', 'hadoop', 'spark', 'hive', 'pig', 'kafka', 'storm', 'flink',
    'softwareengineering', 'systemdesign', 'architecture', 'microservices', 'monolithic', 'soa',
    'ux', 'ui', 'userexperience', 'userinterface', 'design', 'prototyping', 'wireframing', 'figma', 'sketch', 'adobexd',
    'blockchain', 'bitcoin', 'ethereum', 'smartcontracts', 'solidity', 'dapps',
    'gamedevelopment', 'unity', 'unrealengine', 'gamemaker', 'godot',
    'robotics', 'automation', 'iot', 'internetofthings', 'raspberrypi', 'arduino',
    'quantumcomputing', 'quantummechanics', 'quantumalgorithms', 'qubits',
    'bioinformatics', 'biotech', 'genomics', 'proteomics', 'systemsbiology',
    'businessintelligence', 'bi', 'tableau', 'powerbi', 'qlikview', 'looker',
    'crm', 'salesforce', 'hubspot', 'zoho', 'microsoftdynamics',
    'erp', 'sap', 'oracleerp', 'microsoftdynamics365',
    'cms', 'wordpress', 'joomla', 'drupal', 'contentful', 'strapi', 'mobile', 'programming', 'aws',
    'beginners', 'data', 'developer', 'vlsi'
])


#df['keywords_array'] = df['keywords_array'].apply(lambda x: eval(x))
df['keywords_string'] = df['keywords_array'].apply(lambda x: ' '.join(x))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['keywords_string'])
feature_names = vectorizer.get_feature_names_out()


tfidf_scores = defaultdict(lambda: defaultdict(float))
for i, row in enumerate(tfidf_matrix):
    for j, score in zip(row.indices, row.data):
        tfidf_scores[df['id'][i]][feature_names[j]] = score

def extract_top_keywords(keywords, dev_keywords, tfidf_score_dict, top_n=5):

    top_keywords = [kw for kw in keywords if kw in dev_keywords]
    

    if len(top_keywords) < top_n:
        remaining_keywords = [kw for kw in keywords if kw not in top_keywords]
        remaining_keywords = sorted(remaining_keywords, key=lambda x: tfidf_score_dict.get(x, 0), reverse=True)
        top_keywords.extend(remaining_keywords[:top_n - len(top_keywords)])
    
    return top_keywords[:top_n]


df['top_keywords'] = df.apply(lambda row: extract_top_keywords(row['keywords_array'], development_keywords, tfidf_scores[row['id']]), axis=1)


top_keywords_df = df['top_keywords'].apply(pd.Series)
top_keywords_df.columns = ['topword1', 'topword2', 'topword3', 'topword4', 'topword5']


df = pd.concat([df, top_keywords_df], axis=1)

df = df[['id', 'topword1', 'topword2', 'topword3', 'topword4', 'topword5']]
df.to_csv("lecture_tag.csv", index=False)