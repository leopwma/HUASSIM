#********************************************************************************
#
#   Copyright (C) 2019 Culham Centre for Fusion Energy,
#   United Kingdom Atomic Energy Authority, Oxfordshire OX14 3DB, UK
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#*******************************************************************************
#
#   Program: HUASSIM: Huang diffuse X-ray scattering simulation based on elastic dipole tensor
#   File: atomic_form_factor.py
#   Version: 1.0
#   Date:    May 2019
#   Author:  Pui-Wai (Leo) MA
#   Contact: Leo.Ma@ukaea.uk
#   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
#
#*******************************************************************************/

import numpy as np

# database copied from http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
# One should cite http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
# Intensity of diffracted intensities, P. J. Brown, A. G. Fox, E. N. Maslen, M. A. O'Keefe and B. T. M. Willis. 
# International Tables for Crystallography (2006). Vol. C, ch. 6.1, pp. 554-595  [ doi:10.1107/97809553602060000600 ]
# Element 	a1	b1	a2	b2	a3	b3	a4	b4	c

database = \
[["H",0.4899180,20.6593000,0.2620030,7.7403900,0.1967670,49.5519000,0.0498790,2.2015900,0.0013050],\
["H1-",0.8976610,53.1368000,0.5656160,15.1870000,0.4158150,186.5760000,0.1169730,3.5670900,0.0023890],\
["He",0.8734000,9.1037000,0.6309000,3.3568000,0.3112000,22.9276000,0.1780000,0.9821000,0.0064000],\
["Li",1.1282000,3.9546000,0.7508000,1.0524000,0.6175000,85.3905000,0.4653000,168.2610000,0.0377000],\
["Li1+",0.6968000,4.6237000,0.7888000,1.9557000,0.3414000,0.6316000,0.1563000,10.0953000,0.0167000],\
["Be",1.5919000,43.6427000,1.1278000,1.8623000,0.5391000,103.4830000,0.7029000,0.5420000,0.0385000],\
["Be2+",6.2603000,0.0027000,0.8849000,0.8313000,0.7993000,2.2758000,0.1647000,5.1146000,-6.1092000],\
["B",2.0545000,23.2185000,1.3326000,1.0210000,1.0979000,60.3498000,0.7068000,0.1403000,-0.1932000],\
["C",2.3100000,20.8439000,1.0200000,10.2075000,1.5886000,0.5687000,0.8650000,51.6512000,0.2156000],\
["Cval",2.2606900,22.6907000,1.5616500,0.6566650,1.0507500,9.7561800,0.8392590,55.5949000,0.2869770],\
["N",12.2126000,0.0057000,3.1322000,9.8933000,2.0125000,28.9975000,1.1663000,0.5826000,-11.5290000],\
["O",3.0485000,13.2771000,2.2868000,5.7011000,1.5463000,0.3239000,0.8670000,32.9089000,0.2508000],\
["O1-",4.1916000,12.8573000,1.6396900,4.1723600,1.5267300,47.0179000,-20.3070000,-0.0140400,21.9412000],\
["F",3.5392000,10.2825000,2.6412000,4.2944000,1.5170000,0.2615000,1.0243000,26.1476000,0.2776000],\
["F1-",3.6322000,5.2775600,3.5105700,14.7353000,1.2606400,0.4422580,0.9407060,47.3437000,0.6533960],\
["Ne",3.9553000,8.4042000,3.1125000,3.4262000,1.4546000,0.2306000,1.1251000,21.7184000,0.3515000],\
["Na",4.7626000,3.2850000,3.1736000,8.8422000,1.2674000,0.3136000,1.1128000,129.4240000,0.6760000],\
["Na1+",3.2565000,2.6671000,3.9362000,6.1153000,1.3998000,0.2001000,1.0032000,14.0390000,0.4040000],\
["Mg",5.4204000,2.8275000,2.1735000,79.2611000,1.2269000,0.3808000,2.3073000,7.1937000,0.8584000],\
["Mg2+",3.4988000,2.1676000,3.8378000,4.7542000,1.3284000,0.1850000,0.8497000,10.1411000,0.4853000],\
["Al",6.4202000,3.0387000,1.9002000,0.7426000,1.5936000,31.5472000,1.9646000,85.0886000,1.1151000],\
["Al3+",4.1744800,1.9381600,3.3876000,4.1455300,1.2029600,0.2287530,0.5281370,8.2852400,0.7067860],\
["Siv",6.2915000,2.4386000,3.0353000,32.3337000,1.9891000,0.6785000,1.5410000,81.6937000,1.1407000],\
["Sival",5.6626900,2.6652000,3.0716400,38.6634000,2.6244600,0.9169460,1.3932000,93.5458000,1.2470700],\
["Si4+",4.4391800,1.6416700,3.2034500,3.4375700,1.1945300,0.2149000,0.4165300,6.6536500,0.7462970],\
["P",6.4345000,1.9067000,4.1791000,27.1570000,1.7800000,0.5260000,1.4908000,68.1645000,1.1149000],\
["S",6.9053000,1.4679000,5.2034000,22.2151000,1.4379000,0.2536000,1.5863000,56.1720000,0.8669000],\
["Cl",11.4604000,0.0104000,7.1964000,1.1662000,6.2556000,18.5194000,1.6455000,47.7784000,-9.5574000],\
["Cl1-",18.2915000,0.0066000,7.2084000,1.1717000,6.5337000,19.5424000,2.3386000,60.4486000,-16.3780000],\
["Ar",7.4845000,0.9072000,6.7723000,14.8407000,0.6539000,43.8983000,1.6442000,33.3929000,1.4445000],\
["K",8.2186000,12.7949000,7.4398000,0.7748000,1.0519000,213.1870000,0.8659000,41.6841000,1.4228000],\
["K1+",7.9578000,12.6331000,7.4917000,0.7674000,6.3590000,-0.0020000,1.1915000,31.9128000,-4.9978000],\
["Ca",8.6266000,10.4421000,7.3873000,0.6599000,1.5899000,85.7484000,1.0211000,178.4370000,1.3751000],\
["Ca2+",15.6348000,-0.0074000,7.9518000,0.6089000,8.4372000,10.3116000,0.8537000,25.9905000,-14.8750000],\
["Sc",9.1890000,9.0213000,7.3679000,0.5729000,1.6409000,136.1080000,1.4680000,51.3531000,1.3329000],\
["Sc3+",13.4008000,0.2985400,8.0273000,7.9629000,1.6594300,-0.2860400,1.5793600,16.0662000,-6.6667000],\
["Ti",9.7595000,7.8508000,7.3558000,0.5000000,1.6991000,35.6338000,1.9021000,116.1050000,1.2807000],\
["Ti2+",9.1142300,7.5243000,7.6217400,0.4575850,2.2793000,19.5361000,0.0878990,61.6558000,0.8971550],\
["Ti3+",17.7344000,0.2206100,8.7381600,7.0471600,5.2569100,-0.1576200,1.9213400,15.9768000,-14.6520000],\
["Ti4+",19.5114000,0.1788470,8.2347300,6.6701800,2.0134100,-0.2926300,1.5208000,12.9464000,-13.2800000],\
["V",10.2971000,6.8657000,7.3511000,0.4385000,2.0703000,26.8938000,2.0571000,102.4780000,1.2199000],\
["V2+",10.1060000,6.8818000,7.3541000,0.4409000,2.2884000,20.3004000,0.0223000,115.1220000,1.2298000],\
["V3+",9.4314100,6.3953500,7.7419000,0.3833490,2.1534300,15.1908000,0.0168650,63.9690000,0.6565650],\
["V5+",15.6887000,0.6790030,8.1420800,5.4013500,2.0308100,9.9727800,-9.5760000,0.9404640,1.7143000],\
["Cr",10.6406000,6.1038000,7.3537000,0.3920000,3.3240000,20.2626000,1.4922000,98.7399000,1.1832000],\
["Cr2+",9.5403400,5.6607800,7.7509000,0.3442610,3.5827400,13.3075000,0.5091070,32.4224000,0.6168980],\
["Cr3+",9.6809000,5.5946300,7.8113600,0.3343930,2.8760300,12.8288000,0.1135750,32.8761000,0.5182750],\
["Mn",11.2819000,5.3409000,7.3573000,0.3432000,3.0193000,17.8674000,2.2441000,83.7543000,1.0896000],\
["Mn2+",10.8061000,5.2796000,7.3620000,0.3435000,3.5268000,14.3430000,0.2184000,41.3235000,1.0874000],\
["Mn3+",9.8452100,4.9179700,7.8719400,0.2943930,3.5653100,10.8171000,0.3236130,24.1281000,0.3939740],\
["Mn4+",9.9625300,4.8485000,7.9705700,0.2833030,2.7606700,10.4852000,0.0544470,27.5730000,0.2518770],\
["Fe",11.7695000,4.7611000,7.3573000,0.3072000,3.5222000,15.3535000,2.3045000,76.8805000,1.0369000],\
["Fe2+",11.0424000,4.6538000,7.3740000,0.3053000,4.1346000,12.0546000,0.4399000,31.2809000,1.0097000],\
["Fe3+",11.1764000,4.6147000,7.3863000,0.3005000,3.3948000,11.6729000,0.0724000,38.5566000,0.9707000],\
["Co",12.2841000,4.2791000,7.3409000,0.2784000,4.0034000,13.5359000,2.3488000,71.1692000,1.0118000],\
["Co2+",11.2296000,4.1231000,7.3883000,0.2726000,4.7393000,10.2443000,0.7108000,25.6466000,0.9324000],\
["Co3+",10.3380000,3.9096900,7.8817300,0.2386680,4.7679500,8.3558300,0.7255910,18.3491000,0.2866670],\
["Ni",12.8376000,3.8785000,7.2920000,0.2565000,4.4438000,12.1763000,2.3800000,66.3421000,1.0341000],\
["Ni2+",11.4166000,3.6766000,7.4005000,0.2449000,5.3442000,8.8730000,0.9773000,22.1626000,0.8614000],\
["Ni3+",10.7806000,3.5477000,7.7586800,0.2231400,5.2274600,7.6446800,0.8471140,16.9673000,0.3860440],\
["Cu",13.3380000,3.5828000,7.1676000,0.2470000,5.6158000,11.3966000,1.6735000,64.8126000,1.1910000],\
["Cu1+",11.9475000,3.3669000,7.3573000,0.2274000,6.2455000,8.6625000,1.5578000,25.8487000,0.8900000],\
["Cu2+",11.8168000,3.3748400,7.1118100,0.2440780,5.7813500,7.9876000,1.1452300,19.8970000,1.1443100],\
["Zn",14.0743000,3.2655000,7.0318000,0.2333000,5.1652000,10.3163000,2.4100000,58.7097000,1.3041000],\
["Zn2+",11.9719000,2.9946000,7.3862000,0.2031000,6.4668000,7.0826000,1.3940000,18.0995000,0.7807000],\
["Ga",15.2354000,3.0669000,6.7006000,0.2412000,4.3591000,10.7805000,2.9623000,61.4135000,1.7189000],\
["Ga3+",12.6920000,2.8126200,6.6988300,0.2278900,6.0669200,6.3644100,1.0066000,14.4122000,1.5354500],\
["Ge",16.0816000,2.8509000,6.3747000,0.2516000,3.7068000,11.4468000,3.6830000,54.7625000,2.1313000],\
["Ge4+",12.9172000,2.5371800,6.7000300,0.2058550,6.0679100,5.4791300,0.8590410,11.6030000,1.4557200],\
["As",16.6723000,2.6345000,6.0701000,0.2647000,3.4313000,12.9479000,4.2779000,47.7972000,2.5310000],\
["Se",17.0006000,2.4098000,5.8196000,0.2726000,3.9731000,15.2372000,4.3543000,43.8163000,2.8409000],\
["Br",17.1789000,2.1723000,5.2358000,16.5796000,5.6377000,0.2609000,3.9851000,41.4328000,2.9557000],\
["Br1-",17.1718000,2.2059000,6.3338000,19.3345000,5.5754000,0.2871000,3.7272000,58.1535000,3.1776000],\
["Kr",17.3555000,1.9384000,6.7286000,16.5623000,5.5493000,0.2261000,3.5375000,39.3972000,2.8250000],\
["Rb",17.1784000,1.7888000,9.6435000,17.3151000,5.1399000,0.2748000,1.5292000,164.9340000,3.4873000],\
["Rb1+",17.5816000,1.7139000,7.6598000,14.7957000,5.8981000,0.1603000,2.7817000,31.2087000,2.0782000],\
["Sr",17.5663000,1.5564000,9.8184000,14.0988000,5.4220000,0.1664000,2.6694000,132.3760000,2.5064000],\
["Sr2+",18.0874000,1.4907000,8.1373000,12.6963000,2.5654000,24.5651000,-34.1930000,-0.0138000,41.4025000],\
["Y",17.7760000,1.4029000,10.2946000,12.8006000,5.7262900,0.1255990,3.2658800,104.3540000,1.9121300],\
["Y3+",17.9268000,1.3541700,9.1531000,11.2145000,1.7679500,22.6599000,-33.1080000,-0.0131900,40.2602000],\
["Zr",17.8765000,1.2761800,10.9480000,11.9160000,5.4173200,0.1176220,3.6572100,87.6627000,2.0692900],\
["Zr4+",18.1668000,1.2148000,10.0562000,10.1483000,1.0111800,21.6054000,-2.6479000,-0.1027600,9.4145400],\
["Nb",17.6142000,1.1886500,12.0144000,11.7660000,4.0418300,0.2047850,3.5334600,69.7957000,3.7559100],\
["Nb3+",19.8812000,0.0191750,18.0653000,1.1330500,11.0177000,10.1621000,1.9471500,28.3389000,-12.9120000],\
["Nb5+",17.9163000,1.1244600,13.3417000,0.0287810,10.7990000,9.2820600,0.3379050,25.7228000,-6.3934000],\
["Mo",3.7025000,0.2772000,17.2356000,1.0958000,12.8876000,11.0040000,3.7429000,61.6584000,4.3875000],\
["Mo3+",21.1664000,0.0147340,18.2017000,1.0303100,11.7423000,9.5365900,2.3095100,26.6307000,-14.4210000],\
["Mo5+",21.0149000,0.0143450,18.0992000,1.0223800,11.4632000,8.7880900,0.7406250,23.3452000,-14.3160000],\
["Mo6+",17.8871000,1.0364900,11.1750000,8.4806100,6.5789100,0.0588810,0.0000000,0.0000000,0.3449410],\
["Tc",19.1301000,0.8641320,11.0948000,8.1448700,4.6490100,21.5707000,2.7126300,86.8472000,5.4042800],\
["Ru",19.2674000,0.8085200,12.9182000,8.4346700,4.8633700,24.7997000,1.5675600,94.2928000,5.3787400],\
["Ru3+",18.5638000,0.8473290,13.2885000,8.3716400,9.3260200,0.0176620,3.0096400,22.8870000,-3.1892000],\
["Ru4+",18.5003000,0.8445820,13.1787000,8.1253400,4.7130400,0.3649500,2.1853500,20.8504000,1.4235700],\
["Rh",19.2957000,0.7515360,14.3501000,8.2175800,4.7342500,25.8749000,1.2891800,98.6062000,5.3280000],\
["Rh3+",18.8785000,0.7642520,14.1259000,7.8443800,3.3251500,21.2487000,-6.1989000,-0.0103600,11.8678000],\
["Rh4+",18.8545000,0.7608250,13.9806000,7.6243600,2.5346400,19.3317000,-5.6526000,-0.0102000,11.2835000],\
["Pd",19.3319000,0.6986550,15.5017000,7.9892900,5.2953700,25.2052000,0.6058440,76.8986000,5.2659300],\
["Pd2+",19.1701000,0.6962190,15.2096000,7.5557300,4.3223400,22.5057000,0.0000000,0.0000000,5.2916000],\
["Pd4+",19.2493000,0.6838390,14.7900000,7.1483300,2.8928900,17.9144000,-7.9492000,0.0051270,13.0174000],\
["Ag",19.2808000,0.6446000,16.6885000,7.4726000,4.8045000,24.6605000,1.0463000,99.8156000,5.1790000],\
["Ag1+",19.1812000,0.6461790,15.9719000,7.1912300,5.2747500,21.7326000,0.3575340,66.1147000,5.2157200],\
["Ag2+",19.1643000,0.6456430,16.2456000,7.1854400,4.3709000,21.4072000,0.0000000,0.0000000,5.2140400],\
["Cd",19.2214000,0.5946000,17.6444000,6.9089000,4.4610000,24.7008000,1.6029000,87.4825000,5.0694000],\
["Cd2+",19.1514000,0.5979220,17.2535000,6.8063900,4.4712800,20.2521000,0.0000000,0.0000000,5.1193700],\
["In",19.1624000,0.5476000,18.5596000,6.3776000,4.2948000,25.8499000,2.0396000,92.8029000,4.9391000],\
["In3+",19.1045000,0.5515220,18.1108000,6.3247000,3.7889700,17.3595000,0.0000000,0.0000000,4.9963500],\
["Sn",19.1889000,5.8303000,19.1005000,0.5031000,4.4585000,26.8909000,2.4663000,83.9571000,4.7821000],\
["Sn2+",19.1094000,0.5036000,19.0548000,5.8378000,4.5648000,23.3752000,0.4870000,62.2061000,4.7861000],\
["Sn4+",18.9333000,5.7640000,19.7131000,0.4655000,3.4182000,14.0049000,0.0193000,-0.7583000,3.9182000],\
["Sb",19.6418000,5.3034000,19.0455000,0.4607000,5.0371000,27.9074000,2.6827000,75.2825000,4.5909000],\
["Sb3+",18.9755000,0.4671960,18.9330000,5.2212600,5.1078900,19.5902000,0.2887530,55.5113000,4.6962600],\
["Sb5+",19.8685000,5.4485300,19.0302000,0.4679730,2.4125300,14.1259000,0.0000000,0.0000000,4.6926300],\
["Te",19.9644000,4.8174200,19.0138000,0.4208850,6.1448700,28.5284000,2.5239000,70.8403000,4.3520000],\
["I",20.1472000,4.3470000,18.9949000,0.3814000,7.5138000,27.7660000,2.2735000,66.8776000,4.0712000],\
["I1-",20.2332000,4.3579000,18.9970000,0.3815000,7.8069000,29.5259000,2.8868000,84.9304000,4.0714000],\
["Xe",20.2933000,3.9282000,19.0298000,0.3440000,8.9767000,26.4659000,1.9900000,64.2658000,3.7118000],\
["Cs",20.3892000,3.5690000,19.1062000,0.3107000,10.6620000,24.3879000,1.4953000,213.9040000,3.3352000],\
["Cs1+",20.3524000,3.5520000,19.1278000,0.3086000,10.2821000,23.7128000,0.9615000,59.4565000,3.2791000],\
["Ba",20.3361000,3.2160000,19.2970000,0.2756000,10.8880000,20.2073000,2.6959000,167.2020000,2.7731000],\
["Ba2+",20.1807000,3.2136700,19.1136000,0.2833100,10.9054000,20.0558000,0.7763400,51.7460000,3.0290200],\
["La",20.5780000,2.9481700,19.5990000,0.2444750,11.3727000,18.7726000,3.2871900,133.1240000,2.1467800],\
["La3+",20.2489000,2.9207000,19.3763000,0.2506980,11.6323000,17.8211000,0.3360480,54.9453000,2.4086000],\
["Ce",21.1671000,2.8121900,19.7695000,0.2268360,11.8513000,17.6083000,3.3304900,127.1130000,1.8626400],\
["Ce3+",20.8036000,2.7769100,19.5590000,0.2315400,11.9369000,16.5408000,0.6123760,43.1692000,2.0901300],\
["Ce4+",20.3235000,2.6594100,19.8186000,0.2188500,12.1233000,15.7992000,0.1445830,62.2355000,1.5918000],\
["Pr",22.0440000,2.7739300,19.6697000,0.2220870,12.3856000,16.7669000,2.8242800,143.6440000,2.0583000],\
["Pr3+",21.3727000,2.6452000,19.7491000,0.2142990,12.1329000,15.3230000,0.9751800,36.4065000,1.7713200],\
["Pr4+",20.9413000,2.5446700,20.0539000,0.2024810,12.4668000,14.8137000,0.2966890,45.4643000,1.2428500],\
["Nd",22.6845000,2.6624800,19.6847000,0.2106280,12.7740000,15.8850000,2.8513700,137.9030000,1.9848600],\
["Nd3+",21.9610000,2.5272200,19.9339000,0.1992370,12.1200000,14.1783000,1.5103100,30.8717000,1.4758800],\
["Pm",23.3405000,2.5627000,19.6095000,0.2020880,13.1235000,15.1009000,2.8751600,132.7210000,2.0287600],\
["Pm3+",22.5527000,2.4174000,20.1108000,0.1857690,12.0671000,13.1275000,2.0749200,27.4491000,1.1949900],\
["Sm",24.0042000,2.4727400,19.4258000,0.1964510,13.4396000,14.3996000,2.8960400,128.0070000,2.2096300],\
["Sm3+",23.1504000,2.3164100,20.2599000,0.1740810,11.9202000,12.1571000,2.7148800,24.8242000,0.9545860],\
["Eu",24.6274000,2.3879000,19.0886000,0.1942000,13.7603000,13.7546000,2.9227000,123.1740000,2.5745000],\
["Eu2+",24.0063000,2.2778300,19.9504000,0.1735300,11.8034000,11.6096000,3.8724300,26.5156000,1.3638900],\
["Eu3+",23.7497000,2.2225800,20.3745000,0.1639400,11.8509000,11.3110000,3.2650300,22.9966000,0.7593440],\
["Gd",25.0709000,2.2534100,19.0798000,0.1819510,13.8518000,12.9331000,3.5454500,101.3980000,2.4196000],\
["Gd3+",24.3466000,2.1355300,20.4208000,0.1555250,11.8708000,10.5782000,3.7149000,21.7029000,0.6450890],\
["Tb",25.8976000,2.2425600,18.2185000,0.1961430,14.3167000,12.6648000,2.9535400,115.3620000,3.5832400],\
["Tb3+",24.9559000,2.0560100,20.3271000,0.1495250,12.2471000,10.0499000,3.7730000,21.2773000,0.6919670],\
["Dy",26.5070000,2.1802000,17.6383000,0.2021720,14.5596000,12.1899000,2.9657700,111.8740000,4.2972800],\
["Dy3+",25.5395000,1.9804000,20.2861000,0.1433840,11.9812000,9.3497200,4.5007300,19.5810000,0.6896900],\
["Ho",26.9049000,2.0705100,17.2940000,0.1979400,14.5583000,11.4407000,3.6383700,92.6566000,4.5679600],\
["Ho3+",26.1296000,1.9107200,20.0994000,0.1393580,11.9788000,8.8001800,4.9367600,18.5908000,0.8527950],\
["Er",27.6563000,2.0735600,16.4285000,0.2235450,14.9779000,11.3604000,2.9823300,105.7030000,5.9204600],\
["Er3+",26.7220000,1.8465900,19.7748000,0.1372900,12.1506000,8.3622500,5.1737900,17.8974000,1.1761300],\
["Tm",28.1819000,2.0285900,15.8851000,0.2388490,15.1542000,10.9975000,2.9870600,102.9610000,6.7562100],\
["Tm3+",27.3083000,1.7871100,19.3320000,0.1369740,12.3339000,7.9677800,5.3834800,17.2922000,1.6392900],\
["Yb",28.6641000,1.9889000,15.4345000,0.2571190,15.3087000,10.6647000,2.9896300,100.4170000,7.5667200],\
["Yb2+",28.1209000,1.7850300,17.6817000,0.1599700,13.3335000,8.1830400,5.1465700,20.3900000,3.7098300],\
["Yb3+",27.8917000,1.7327200,18.7614000,0.1387900,12.6072000,7.6441200,5.4764700,16.8153000,2.2600100],\
["Lu",28.9476000,1.9018200,15.2208000,9.9851900,15.1000000,0.2610330,3.7160100,84.3298000,7.9762800],\
["Lu3+",28.4628000,1.6821600,18.1210000,0.1422920,12.8429000,7.3372700,5.5941500,16.3535000,2.9757300],\
["Hf",29.1440000,1.8326200,15.1726000,9.5999000,14.7586000,0.2751160,4.3001300,72.0290000,8.5815400],\
["Hf4+",28.8131000,1.5913600,18.4601000,0.1289030,12.7285000,6.7623200,5.5992700,14.0366000,2.3969900],\
["Ta",29.2024000,1.7733300,15.2293000,9.3704600,14.5135000,0.2959770,4.7649200,63.3644000,9.2435400],\
["Ta5+",29.1587000,1.5071100,18.8407000,0.1167410,12.8268000,6.3152400,5.3869500,12.4244000,1.7855500],\
["W",29.0818000,1.7202900,15.4300000,9.2259000,14.4327000,0.3217030,5.1198200,57.0560000,9.8875000],\
["W6+",29.4936000,1.4275500,19.3763000,0.1046210,13.0544000,5.9366700,5.0641200,11.1972000,1.0107400],\
["Re",28.7621000,1.6719100,15.7189000,9.0922700,14.5564000,0.3505000,5.4417400,52.0861000,10.4720000],\
["Os",28.1894000,1.6290300,16.1550000,8.9794800,14.9305000,0.3826610,5.6758900,48.1647000,11.0005000],\
["Os4+",30.4190000,1.3711300,15.2637000,6.8470600,14.7458000,0.1651910,5.0679500,18.0030000,6.4980400],\
["Ir",27.3049000,1.5927900,16.7296000,8.8655300,15.6115000,0.4179160,5.8337700,45.0011000,11.4722000],\
["Ir3+",30.4156000,1.3432300,15.8620000,7.1090900,13.6145000,0.2046330,5.8200800,20.3254000,8.2790300],\
["Ir4+",30.7058000,1.3092300,15.5512000,6.7198300,14.2326000,0.1672520,5.5367200,17.4911000,6.9682400],\
["Pt",27.0059000,1.5129300,17.7639000,8.8117400,15.7131000,0.4245930,5.7837000,38.6103000,11.6883000],\
["Pt2+",29.8429000,1.3292700,16.7224000,7.3897900,13.2153000,0.2632970,6.3523400,22.9426000,9.8532900],\
["Pt4+",30.9612000,1.2481300,15.9829000,6.6083400,13.7348000,0.1686400,5.9203400,16.9392000,7.3953400],\
["Au",16.8819000,0.4611000,18.5913000,8.6216000,25.5582000,1.4826000,5.8600000,36.3956000,12.0658000],\
["Au1+",28.0109000,1.3532100,17.8204000,7.7395000,14.3359000,0.3567520,6.5807700,26.4043000,11.2299000],\
["Au3+",30.6886000,1.2199000,16.9029000,6.8287200,12.7801000,0.2128670,6.5235400,18.6590000,9.0968000],\
["Hg",20.6809000,0.5450000,19.0417000,8.4484000,21.6575000,1.5729000,5.9676000,38.3246000,12.6089000],\
["Hg1+",25.0853000,1.3950700,18.4973000,7.6510500,16.8883000,0.4433780,6.4821600,28.2262000,12.0205000],\
["Hg2+",29.5641000,1.2115200,18.0600000,7.0563900,12.8374000,0.2847380,6.8991200,20.7482000,10.6268000],\
["Tl",27.5446000,0.6551500,19.1584000,8.7075100,15.5380000,1.9634700,5.5259300,45.8149000,13.1746000],\
["Tl1+",21.3985000,1.4711000,20.4723000,0.5173940,18.7478000,7.4346300,6.8284700,28.8482000,12.5258000],\
["Tl3+",30.8695000,1.1008000,18.3481000,6.5385200,11.9328000,0.2190740,7.0057400,17.2114000,9.8027000],\
["Pb",31.0617000,0.6902000,13.0637000,2.3576000,18.4420000,8.6180000,5.9696000,47.2579000,13.4118000],\
["Pb2+",21.7886000,1.3366000,19.5682000,0.4883830,19.1406000,6.7727000,7.0110700,23.8132000,12.4734000],\
["Pb4+",32.1244000,1.0056600,18.8003000,6.1092600,12.0175000,0.1470410,6.9688600,14.7140000,8.0842800],\
["Bi",33.3689000,0.7040000,12.9510000,2.9238000,16.5877000,8.7937000,6.4692000,48.0093000,13.5782000],\
["Bi3+",21.8053000,1.2356000,19.5026000,6.2414900,19.1053000,0.4699990,7.1029500,20.3185000,12.4711000],\
["Bi5+",33.5364000,0.9165400,25.0946000,0.3904200,19.2497000,5.7141400,6.9155500,12.8285000,-6.7994000],\
["Po",34.6726000,0.7009990,15.4733000,3.5507800,13.1138000,9.5564200,7.0258800,47.0045000,13.6770000],\
["At",35.3163000,0.6858700,19.0211000,3.9745800,9.4988700,11.3824000,7.4251800,45.4715000,13.7108000],\
["Rn",35.5631000,0.6631000,21.2816000,4.0691000,8.0037000,14.0422000,7.4433000,44.2473000,13.6905000],\
["Fr",35.9299000,0.6464530,23.0547000,4.1761900,12.1439000,23.1052000,2.1125300,150.6450000,13.7247000],\
["Ra",35.7630000,0.6163410,22.9064000,3.8713500,12.4739000,19.9887000,3.2109700,142.3250000,13.6211000],\
["Ra2+",35.2150000,0.6049090,21.6700000,3.5767000,7.9134200,12.6010000,7.6507800,29.8436000,13.5431000],\
["Ac",35.6597000,0.5890920,23.1032000,3.6515500,12.5977000,18.5990000,4.0865500,117.0200000,13.5266000],\
["Ac3+",35.1736000,0.5796890,22.1112000,3.4143700,8.1921600,12.9187000,7.0554500,25.9443000,13.4637000],\
["Th",35.5645000,0.5633590,23.4219000,3.4620400,12.7473000,17.8309000,4.8070300,99.1722000,13.4314000],\
["Th4+",35.1007000,0.5550540,22.4418000,3.2449800,9.7855400,13.4661000,5.2944400,23.9533000,13.3760000],\
["Pa",35.8847000,	0.5477510,23.2948000,3.4151900,14.1891000,16.9235000,4.1728700,105.2510000,13.4287000],\
["U",36.0228000,0.5293000,23.4128000,3.3253000,14.9491000,16.0927000,4.1880000,100.6130000,13.3966000],\
["U3+",35.5747000,0.5204800,22.5259000,3.1229300,12.2165000,12.7148000,5.3707300,26.3394000,13.3092000],\
["U4+",35.3715000,0.5165980,22.5326000,3.0505300,12.0291000,12.5723000,4.7984000,23.4582000,13.2671000],\
["U6+",34.8509000,0.5070790,22.7584000,2.8903000,14.0099000,13.1767000,1.2145700,25.2017000,13.1665000],\
["Np",36.1874000,0.5119290,23.5964000,3.2539600,15.6402000,15.3622000,4.1855000,97.4908000,13.3573000],\
["Np3+",35.7074000,0.5023220,22.6130000,3.0380700,12.9898000,12.1449000,5.4322700,25.4928000,13.2544000],\
["Np4+",35.5103000,0.4986260,22.5787000,2.9662700,12.7766000,11.9484000,4.9215900,22.7502000,13.2116000],\
["Np6+",35.0136000,0.4898100,22.7286000,2.8109900,14.3884000,12.3300000,1.7566900,22.6581000,13.1130000],\
["Pu",36.5254000,0.4993840,23.8083000,3.2637100,16.7707000,14.9455000,3.4794700,105.9800000,13.3812000],\
["Pu3+",35.8400000,0.4849380,22.7169000,2.9611800,13.5807000,11.5331000,5.6601600,24.3992000,13.1991000],\
["Pu4+",35.6493000,0.4814220,22.6460000,2.8902000,13.3595000,11.3160000,5.1883100,21.8301000,13.1555000],\
["Pu6+",35.1736000,0.4732040,22.7181000,2.7384800,14.7635000,11.5530000,2.2867800,20.9303000,13.0582000],\
["Am",36.6706000,0.4836290,24.0992000,3.2064700,17.3415000,14.3136000,3.4933100,102.2730000,13.3592000],\
["Cm",36.6488000,0.4651540,24.4096000,3.0899700,17.3990000,13.4346000,4.2166500,88.4834000,13.2887000],\
["Bk",36.7881000,0.4510180,24.7736000,3.0461900,17.8919000,12.8946000,4.2328400,86.0030000,13.2754000],\
["Cf",36.9185000,0.4375330,25.1995000,3.0077500,18.3317000,12.4044000,4.2439100,83.7881000,13.2674000]]

def atomic_form_factor(q, element):

    in_database = 0
    a = np.zeros(4)
    b = np.zeros(4)
    c = 0.0
    f = 0.0

    # q is the scattering vector, in unit of angstrom^-1
    for x in database:
        if x[0] == element:
            a[0] = float(x[1])
            b[0] = float(x[2])
            a[1] = float(x[3])
            b[1] = float(x[4])
            a[2] = float(x[5])
            b[2] = float(x[6])
            a[3] = float(x[7])
            b[3] = float(x[8])
            c    = float(x[9])
            for i in range(4):
                f += a[i]*np.exp(-b[i]*(q/(4e0*np.pi))**2) 
            f += c
            in_database = 1
    if in_database == 0:
        print ("Not an element on database. Setting atomic form factor f = 1.0.")
        f = 1.0

    return f
             
       
