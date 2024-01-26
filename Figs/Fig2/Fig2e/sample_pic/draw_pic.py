import matplotlib.pyplot as plt


con_list = []
mu_list = []
con_data = open("./pearson_con.txt")
for line in con_data:
    line = line.rstrip().split()
    con_list.append(float(line[0]))

con_data.close()

drug_index = []
drug_corr = []
for i in range(len(con_list)):
    if con_list[i] > 0.610:
        drug_index.append(i)
        drug_corr.append(con_list[i])

drug_list = []
drug_data = open("./drug2ind.txt")
for line in drug_data:
    line = line.rstrip().split()
    if int(line[0]) in drug_index:
        drug_list.append(line[1])
drug_data.close()

max_i_list = []
for epoch in range(12):
    max_i = 0
    max_v = 0

    for i in range(len(drug_corr)):
        if len(drug_corr) == 1:
            max_i_list.append(i)
        if drug_corr[i] > max_v and i not in max_i_list:
            max_v = drug_corr[i]
            max_i = i
    max_i_list.append(max_i)

with open("top_drug.txt", "w") as top:
    for i in range(len(max_i_list)):
        top.write(drug_list[max_i_list[i]] + "\n")
top.close()


sort_con_list_bar = sorted(con_list, reverse=True)
sort_con_list = sorted(drug_corr, reverse=True)
sort_con_list.remove(sort_con_list[1])
sort_con_list.remove(sort_con_list[6])

plt.plot([0.0, 684], [0.0, 0.0], color='black', linewidth=1, label="best line")
bar_color = ['red'] * 58 + ['lime'] * 626
plt.bar(range(len(sort_con_list_bar)), sort_con_list_bar, width=1, color=bar_color)
plt.yticks(fontsize=15)

ax=plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(0)
ax.spines['top'].set_linewidth(0)

plt.savefig('./barplot.png', bbox_inches='tight')
plt.close()

drug_name = ["Teniposide", "Vincristine", "Decitabine", "BRD2889", "GSK461364", "Tivantinib", "Barasertib", "KPT-185", "KX2-391", "Topotecan"]
plt.bar(drug_name, sort_con_list[0:10], width=0.8, color=['red'])
plt.xticks(rotation=70, fontsize=31)
plt.yticks([0.1, 0.3, 0.5, 0.7], [0.1, 0.3, 0.5, 0.7], fontsize=40)

plt.savefig('./top10.png', bbox_inches='tight')

