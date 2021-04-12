# -*- coding: utf-8 -*-
# === pkmap_data.py ===



# artifical Dataset for illustration propuse
AD = {
    '0000': 0, 
    '0001': 0, 
    '0011': 0, 
    '0010': 0, 
    '0110': 0, 
    '0111': 0, 
    '0101': 0, 
    '0100': 0, 
    '1100': 255, 
    '1101': 0, 
    '1111': 604, 
    '1110': 58852, 
    '1010': 2771, 
    '1011': 38, 
    '1001': 40272, 
    '1000': 1282001, 
}


# pre-set pats data: x_data, y_data, width, hight
# for 9 variables scenarios only
pat_data = {
    9:{
        '3(7+8+9)': ((0.5, 16.5), (1.5, 9.5), 14, 4),
        '-3-7-8': ((-2.5, 13.5, 29.5), (-2.5, 5.5, 13.5), 4, 4),
        '-3-4-7-8': ((-2.5, 13.5, 29.5), (-1.5, 6.5, 14.5), 4, 2),
        '-3-7-8-9': ((-1.5, 14.5, 30.5), (-2.5, 5.5, 13.5), 2, 4),
        '4(8+9)': ((0.5,8.5,16.5,24.5), (0.5,4.5,8.5,12.5),6,2), 
    },

}


# mean and std of each appliance
app_data = {
    # house number
	# House 1
	1:{
        # appliance number
        # generated by do2() in _try1.py
        # name, threashold, mean, std
		"Aggregate":{
			"thrd": 1888,       # not used
			"mean": 3426.135924479211,
			"std": 2179.5451107411473,
		},
		"Appliance1":{
			"name": "Fridge",
			"thrd": 38,
			"mean": 77.4257529688486,
			"std": 60.007439647845196,
		},
		"Appliance2":{
			"name": "Freezer(1)",
			"thrd": 24,
			"mean": 48.24484434426202,
			"std": 29.98873172471006,
		},
		"Appliance3":{
			"name": "Freezer(2)",
			"thrd": 36,
			"mean": 72.4325155654045,
			"std": 21.555104352126992,
		},
		"Appliance4":{
			"name": "Washer_Dryer",
			"thrd": 846,
			"mean": 1759.8576185191057,
			"std": 561.331432848893,
		},
		"Appliance5":{
			"name": "Washing_Machine",
			"thrd": 1127,
			"mean": 2274.3004853654948,
			"std": 97.44216899447953,
		},
		"Appliance6":{
			"name": "Dishwasher",
			"thrd": 1133,
			"mean": 2243.807122966437,
			"std": 49.34633069049131,
		},
		"Appliance7":{
			"name": "Computer",
			"thrd": 14,
			"mean": 29.79505506336094,
			"std": 30.32314328930331,
		},
		"Appliance8":{
			"name": "Television_Site",
			"thrd": 16,
			"mean": 32.74688945850847,
			"std": 10.358726245380758,
		},
		"Appliance9":{
			"name": "Electric_Heater",
			"thrd": 507,
			"mean": 1014.7383970642983,
			"std": 102.59462072024293,
		},
	},
	# House 2
	2:{
		"Aggregate":{
			"thrd": 1855,
			"mean": 3483.0889594021246,
			"std": 2277.583095291989,
		},
		"Appliance1":{
			"name": "Fridge-Freezer",
			"thrd": 44,
			"mean": 88.41121831200067,
			"std": 25.359713897773723,
		},
		"Appliance2":{
			"name": "Washing_Machine",
			"thrd": 1028,
			"mean": 2052.772049674287,
			"std": 58.611443042747915,
		},
		"Appliance3":{
			"name": "Dishwasher",
			"thrd": 1084,
			"mean": 2198.382397431748,
			"std": 37.976235506486574,
		},
		"Appliance4":{
			"name": "Television_Site",
			"thrd": 21,
			"mean": 43.74258356512216,
			"std": 15.72508492871061,
		},
		"Appliance5":{
			"name": "Microwave",
			"thrd": 553,
			"mean": 1111.1399332451665,
			"std": 65.72139581991566,
		},
		"Appliance6":{
			"name": "Toaster",
			"thrd": 439,
			"mean": 943.2636125082003,
			"std": 119.54369386798396,
		},
		"Appliance7":{
			"name": "Hi-Fi",
			"thrd": 8,
			"mean": 16.099194146790357,
			"std": 8.363954561324126,
		},
		"Appliance8":{
			"name": "Kettle",
			"thrd": 1352,
			"mean": 2719.715312088274,
			"std": 71.65178391323498,
		},
		"Appliance9":{
			"name": "Overhead_Fan",
			"thrd": 30,
			"mean": 60.56811336450475,
			"std": 33.044661755479076,
		},
	},
	# House 3
	3:{
		"Aggregate":{
			"thrd": 1788,
			"mean": 3150.59495754421,
			"std": 1884.280877493726,
		},
		"Appliance1":{
			"name": "Toaster",
			"thrd": 510,
			"mean": 1020.7943794728574,
			"std": 91.16078197068454,
		},
		"Appliance2":{
			"name": "Fridge-Freezer",
			"thrd": 51,
			"mean": 102.5176587357509,
			"std": 44.26949603436294,
		},
		"Appliance3":{
			"name": "Freezer",
			"thrd": 45,
			"mean": 90.72543538969647,
			"std": 30.803406223623828,
		},
		"Appliance4":{
			"name": "Tumble_Dryer",
			"thrd": 1199,
			"mean": 2396.7946549169255,
			"std": 186.62116984984854,
		},
		"Appliance5":{
			"name": "Dishwasher",
			"thrd": 1067,
			"mean": 2113.156661037894,
			"std": 61.754898092150064,
		},
		"Appliance6":{
			"name": "Washing_Machine",
			"thrd": 936,
			"mean": 1864.991406694085,
			"std": 115.26737262112886,
		},
		"Appliance7":{
			"name": "Television_Site",
			"thrd": 71,
			"mean": 143.70938345805823,
			"std": 12.644094914732252,
		},
		"Appliance8":{
			"name": "Microwave",
			"thrd": 660,
			"mean": 1328.6964827841541,
			"std": 267.84176174881696,
		},
		"Appliance9":{
			"name": "Kettle",
			"thrd": 899,
			"mean": 1812.549643604759,
			"std": 264.76169609687196,
		},
	},
	# House 4
	4:{
		"Aggregate":{
			"thrd": 1494,
			"mean": 2659.1290568205345,
			"std": 1115.1985832436708,
		},
		"Appliance1":{
			"name": "Fridge",
			"thrd": 26,
			"mean": 52.5040534138378,
			"std": 33.647187673025805,
		},
		"Appliance2":{
			"name": "Freezer",
			"thrd": 70,
			"mean": 141.43575758863633,
			"std": 32.10687518451993,
		},
		"Appliance3":{
			"name": "Fridge-Freezer",
			"thrd": 64,
			"mean": 128.9876920486507,
			"std": 27.54122971220632,
		},
		"Appliance4":{
			"name": "Washing_Machine(1)",
			"thrd": 1256,
			"mean": 2518.3547162065074,
			"std": 75.8192808219889,
		},
		"Appliance5":{
			"name": "Washing_Machine(2)",
			"thrd": 1193,
			"mean": 2365.4795356231743,
			"std": 186.69527287523968,
		},
		"Appliance6":{
			"name": "Desktop_Computer",
			"thrd": 49,
			"mean": 91.04729077564814,
			"std": 22.42614528354936,
		},
		"Appliance7":{
			"name": "Television_Site",
			"thrd": 40,
			"mean": 77.14345401228265,
			"std": 11.552825701633013,
		},
		"Appliance8":{
			"name": "Microwave",
			"thrd": 569,
			"mean": 1131.429396256836,
			"std": 70.02603375835034,
		},
		"Appliance9":{
			"name": "Kettle",
			"thrd": 947,
			"mean": 1914.5327095371892,
			"std": 83.88892260378849,
		},
	},
	# House 5
	5:{
		"Aggregate":{
			"thrd": 1553,
			"mean": 2608.3937669954616,
			"std": 1130.749753815313,
		},
		"Appliance1":{
			"name": "Fridge-Freezer",
			"thrd": 59,
			"mean": 116.48822544021846,
			"std": 57.1710323893099,
		},
		"Appliance2":{
			"name": "Tumble_Dryer",
			"thrd": 981,
			"mean": 1927.828133262824,
			"std": 276.382961000236,
		},
		"Appliance3":{
			"name": "Washing_Machine",
			"thrd": 1043,
			"mean": 2076.3162811326247,
			"std": 88.48878261985166,
		},
		"Appliance4":{
			"name": "Dishwasher",
			"thrd": 1159,
			"mean": 2332.0047520624657,
			"std": 93.87638608111334,
		},
		"Appliance5":{
			"name": "Desktop_Computer",
			"thrd": 53,
			"mean": 100.56709740211335,
			"std": 32.893343133659826,
		},
		"Appliance6":{
			"name": "Television_Site",
			"thrd": 38,
			"mean": 77.64343242754427,
			"std": 13.810659117695108,
		},
		"Appliance7":{
			"name": "Microwave",
			"thrd": 892,
			"mean": 1783.9874012385224,
			"std": 563.8664917063825,
		},
		"Appliance8":{
			"name": "Kettle",
			"thrd": 1364,
			"mean": 2727.2090943589237,
			"std": 97.54733616803297,
		},
		"Appliance9":{
			"name": "Toaster",
			"thrd": 752,
			"mean": 1502.8323487200028,
			"std": 513.4277497713133,
		},
	},
	# House 6
	6:{
		"Aggregate":{
			"thrd": 1701,
			"mean": 2987.028861062162,
			"std": 729.2181448987446,
		},
		"Appliance1":{
			"name": "Freezer",
			"thrd": 29,
			"mean": 58.582987220672294,
			"std": 38.52905686739248,
		},
		"Appliance2":{
			"name": "Washing_Machine",
			"thrd": 989,
			"mean": 1998.7986047922354,
			"std": 123.52873716704671,
		},
		"Appliance3":{
			"name": "Dishwasher",
			"thrd": 1108,
			"mean": 2197.449891405523,
			"std": 44.02129409147546,
		},
		"Appliance4":{
			"name": "MJY_Computer",
			"thrd": 92,
			"mean": 174.3375065384003,
			"std": 28.714947512634218,
		},
		"Appliance5":{
			"name": "TV/Satellite",
			"thrd": 65,
			"mean": 119.25643322913538,
			"std": 19.404230753292072,
		},
		"Appliance6":{
			"name": "Microwave",
			"thrd": 700,
			"mean": 1399.589079956572,
			"std": 203.796648890547,
		},
		"Appliance7":{
			"name": "Kettle",
			"thrd": 1306,
			"mean": 2619.481144691094,
			"std": 73.23396988243918,
		},
		"Appliance8":{
			"name": "Toaster",
			"thrd": 513,
			"mean": 1025.0453827940016,
			"std": 215.01148631971665,
		},
		"Appliance9":{
			"name": "PGM_Computer",
			"thrd": 44,
			"mean": 82.50559609616332,
			"std": 43.53375637100229,
		},
	},
	# House 7
	7:{
		"Aggregate":{
			"thrd": 1631,
			"mean": 2925.4651925951284,
			"std": 988.6631491845259,
		},
		"Appliance1":{
			"name": "Fridge",
			"thrd": 40,
			"mean": 79.81231377254618,
			"std": 21.733533411609518,
		},
		"Appliance2":{
			"name": "Freezer(1)",
			"thrd": 49,
			"mean": 96.00677824420444,
			"std": 34.7949760390909,
		},
		"Appliance3":{
			"name": "Freezer(2)",
			"thrd": 661,
			"mean": 1324.9977642925583,
			"std": 462.5529480042907,
		},
		"Appliance4":{
			"name": "Tumble_Dryer",
			"thrd": 1479,
			"mean": 2950.843681632856,
			"std": 616.6716514479951,
		},
		"Appliance5":{
			"name": "Washing_Machine",
			"thrd": 1022,
			"mean": 2036.2895323383084,
			"std": 84.96774375995112,
		},
		"Appliance6":{
			"name": "Dishwasher",
			"thrd": 1075,
			"mean": 2146.1349020960383,
			"std": 73.72347116898368,
		},
		"Appliance7":{
			"name": "Television_Site",
			"thrd": 22,
			"mean": 43.927796822247934,
			"std": 10.653720791306439,
		},
		"Appliance8":{
			"name": "Toaster",
			"thrd": 462,
			"mean": 922.7132920389927,
			"std": 87.70473786476474,
		},
		"Appliance9":{
			"name": "Kettle",
			"thrd": 1102,
			"mean": 2210.8280744787603,
			"std": 76.18767848609852,
		},
	},
	# House 8
	8:{
		"Aggregate":{
			"thrd": 2035,
			"mean": 3678.098145234926,
			"std": 2084.152644322654,
		},
		"Appliance1":{
			"name": "Fridge",
			"thrd": 576,
			"mean": 1144.7455540355677,
			"std": 401.07865171653276,
		},
		"Appliance2":{
			"name": "Freezer",
			"thrd": 40,
			"mean": 81.26034731249474,
			"std": 56.43021752867993,
		},
		"Appliance3":{
			"name": "Washer_Dryer",
			"thrd": 941,
			"mean": 1874.4390088945363,
			"std": 297.75609379720294,
		},
		"Appliance4":{
			"name": "Washing_Machine",
			"thrd": 1156,
			"mean": 2262.7503361909567,
			"std": 103.87818590876778,
		},
		"Appliance5":{
			"name": "Toaster",
			"thrd": 453,
			"mean": 882.4031617591968,
			"std": 172.2155905249411,
		},
		"Appliance6":{
			"name": "Computer",
			"thrd": 55,
			"mean": 93.4367311586315,
			"std": 49.8677980777319,
		},
		"Appliance7":{
			"name": "Television_Site",
			"thrd": 74,
			"mean": 143.19486377428876,
			"std": 26.571211726050056,
		},
		"Appliance8":{
			"name": "Microwave",
			"thrd": 724,
			"mean": 1446.6342634535242,
			"std": 213.3690432355664,
		},
		"Appliance9":{
			"name": "Kettle",
			"thrd": 1381,
			"mean": 2743.703128731789,
			"std": 81.96373467235757,
		},
	},
	# House 9
	9:{
		"Aggregate":{
			"thrd": 1897,
			"mean": 3457.4134976488444,
			"std": 1645.8954456559427,
		},
		"Appliance1":{
			"name": "Fridge-Freezer",
			"thrd": 47,
			"mean": 94.11927138708603,
			"std": 60.307583048627876,
		},
		"Appliance2":{
			"name": "Washer_Dryer",
			"thrd": 846,
			"mean": 1687.1213081166272,
			"std": 474.7625167524102,
		},
		"Appliance3":{
			"name": "Washing_Machine",
			"thrd": 1086,
			"mean": 2145.8963414634145,
			"std": 114.54209693652018,
		},
		"Appliance4":{
			"name": "Dishwasher",
			"thrd": 1097,
			"mean": 2182.9072627360115,
			"std": 57.995090056907166,
		},
		"Appliance5":{
			"name": "Television_Site",
			"thrd": 56,
			"mean": 103.5728398629294,
			"std": 13.732483741107824,
		},
		"Appliance6":{
			"name": "Microwave",
			"thrd": 569,
			"mean": 1095.0773703571851,
			"std": 111.8337621551064,
		},
		"Appliance7":{
			"name": "Kettle",
			"thrd": 1361,
			"mean": 2726.728486526093,
			"std": 93.46063316718602,
		},
		"Appliance8":{
			"name": "Hi-Fi",
			"thrd": 8,
			"mean": 11.104060717996923,
			"std": 2.5150648749881417,
		},
		"Appliance9":{
			"name": "Electric_Heater",
			"thrd": 934,
			"mean": 1952.791665001798,
			"std": 52.60735456709494,
		},
	},
	# House 10
	10:{
		"Aggregate":{
			"thrd": 1541,
			"mean": 2625.9548861958747,
			"std": 848.6477689289907,
		},
		"Appliance1":{
			"name": "Magimix(Blender)",
			"thrd": 38,
			"mean": 77.85831646015981,
			"std": 14.883846425840844,
		},
		"Appliance2":{
			"name": "Toaster",
			"thrd": 995,
			"mean": 1934.0125,
			"std": 385.0782999911853,
		},
		"Appliance3":{
			"name": "Chest_Freezer",
			"thrd": 35,
			"mean": 71.28331157944068,
			"std": 19.818772021571593,
		},
		"Appliance4":{
			"name": "Fridge-Freezer",
			"thrd": 53,
			"mean": 101.61680233548422,
			"std": 32.71343599339202,
		},
		"Appliance5":{
			"name": "Washing_Machine",
			"thrd": 1009,
			"mean": 1985.7448195546779,
			"std": 116.91884228656197,
		},
		"Appliance6":{
			"name": "Dishwasher",
			"thrd": 853,
			"mean": 1743.5554275665459,
			"std": 66.86867461747113,
		},
		"Appliance7":{
			"name": "Television_Site",
			"thrd": 71,
			"mean": 99.85710615381434,
			"std": 21.174522638192446,
		},
		"Appliance8":{
			"name": "Microwave",
			"thrd": 583,
			"mean": 1166.6094336354713,
			"std": 55.35189172855603,
		},
		"Appliance9":{
			"name": "K_Mix",
			"thrd": 967,
			"mean": 1887.8712138465041,
			"std": 76.4473043100826,
		},
	},
	# House 11
	11:{
		"Aggregate":{
			"thrd": 1062,
			"mean": 1869.9679997588235,
			"std": 571.3210628293123,
		},
		"Appliance1":{
			"name": "Fridge",
			"thrd": 1155,
			"mean": 2237.9332298136646,
			"std": 344.72626603098064,
		},
		"Appliance2":{
			"name": "Fridge-Freezer",
			"thrd": 41,
			"mean": 82.36003690221258,
			"std": 24.038414634970064,
		},
		"Appliance3":{
			"name": "Washing_Machine",
			"thrd": 1124,
			"mean": 2287.1860093803016,
			"std": 60.748262732373746,
		},
		"Appliance4":{
			"name": "Dishwasher",
			"thrd": 1091,
			"mean": 2248.077923357305,
			"std": 455.2521498435832,
		},
		"Appliance5":{
			"name": "Computer_Site",
			"thrd": 18,
			"mean": 34.998164147807856,
			"std": 18.0283984161216,
		},
		"Appliance6":{
			"name": "Microwave",
			"thrd": 561,
			"mean": 1105.6196933010492,
			"std": 74.2711958022679,
		},
		"Appliance7":{
			"name": "Kettle",
			"thrd": 1044,
			"mean": 2083.197803631619,
			"std": 48.818306975012455,
		},
		"Appliance8":{
			"name": "Router",
			"thrd": 11,
			"mean": 12.21671160218306,
			"std": 0.5135205065793328,
		},
		"Appliance9":{
			"name": "Hi-Fi",
			"thrd": 9,
			"mean": 19.86143855844005,
			"std": 6.866999391386706,
		},
	},
	# House 12
	12:{
		"Aggregate":{
			"thrd": 1475,
			"mean": 2725.031046013709,
			"std": 742.5698359030744,
		},
		"Appliance1":{
			"name": "Fridge-Freezer",
			"thrd": 46,
			"mean": 93.17640392827707,
			"std": 28.202391869928785,
		},
		"Appliance2":{
			"name": "Unknown",
			"thrd": 93,
			"mean": 179.15960691291087,
			"std": 32.052454086309474,
		},
		"Appliance3":{
			"name": "Unknown",
			"thrd": 702,
			"mean": 1415.9994839042643,
			"std": 151.88923808122456,
		},
		"Appliance4":{
			"name": "Computer_Site",
			"thrd": 1394,
			"mean": 2802.5381543921917,
			"std": 77.55499621478067,
		},
		"Appliance5":{
			"name": "Microwave",
			"thrd": 459,
			"mean": 937.2067658998646,
			"std": 88.90224799424803,
		},
		"Appliance6":{
			"name": "Kettle",
			"thrd": 56,
			"mean": 113.22308378613303,
			"std": 13.6845965365353,
		},
		"Appliance7":{
			"name": "Toaster",
			"thrd": 0,
			"mean": 0.0,
			"std": 0.0,
		},
		"Appliance8":{
			"name": "Television",
			"thrd": 0,
			"mean": 0.0,
			"std": 0.0,
		},
		"Appliance9":{
			"name": "Unknown",
			"thrd": 0,
			"mean": 0.0,
			"std": 0.0,
		},
	},
	# House 13
	13:{
		"Aggregate":{
			"thrd": 1660,
			"mean": 2972.724404876594,
			"std": 1070.1812560339108,
		},
		"Appliance1":{
			"name": "Television_Site",
			"thrd": 89,
			"mean": 179.33310071952195,
			"std": 72.06479150151965,
		},
		"Appliance2":{
			"name": "Freezer",
			"thrd": 803,
			"mean": 1529.8335483342548,
			"std": 38.90374789827551,
		},
		"Appliance3":{
			"name": "Washing_Machine",
			"thrd": 1047,
			"mean": 2088.8406670582526,
			"std": 89.09898396591119,
		},
		"Appliance4":{
			"name": "Dishwasher",
			"thrd": 855,
			"mean": 1868.389038697251,
			"std": 43.74048793580965,
		},
		"Appliance5":{
			"name": "Unknown",
			"thrd": 1184,
			"mean": 2362.642267503552,
			"std": 496.3991872402785,
		},
		"Appliance6":{
			"name": "Network_Site",
			"thrd": 70,
			"mean": 140.38747094206414,
			"std": 35.33824729556349,
		},
		"Appliance7":{
			"name": "Microwave",
			"thrd": 30,
			"mean": 52.96664355975513,
			"std": 21.58683737405849,
		},
		"Appliance8":{
			"name": "Microwave",
			"thrd": 684,
			"mean": 1433.3008300494675,
			"std": 59.108023103912046,
		},
		"Appliance9":{
			"name": "Kettle",
			"thrd": 1266,
			"mean": 2551.5442900841194,
			"std": 62.05662553301707,
		},
	},
	# House 15
	15:{
		"Aggregate":{
			"thrd": 1520,
			"mean": 2826.201751058529,
			"std": 666.1191519637356,
		},
		"Appliance1":{
			"name": "Fridge-Freezer",
			"thrd": 39,
			"mean": 78.13977824674237,
			"std": 25.583433270248044,
		},
		"Appliance2":{
			"name": "Tumble_Dryer",
			"thrd": 1234,
			"mean": 2463.1315289825397,
			"std": 315.12183157979064,
		},
		"Appliance3":{
			"name": "Washing_Machine",
			"thrd": 1074,
			"mean": 2145.6419281175536,
			"std": 103.98508436250236,
		},
		"Appliance4":{
			"name": "Dishwasher",
			"thrd": 960,
			"mean": 2037.6414835164835,
			"std": 111.31218787760241,
		},
		"Appliance5":{
			"name": "Computer_Site",
			"thrd": 10,
			"mean": 20.987194059525887,
			"std": 18.438783978049305,
		},
		"Appliance6":{
			"name": "Television_Site",
			"thrd": 34,
			"mean": 69.8591918956152,
			"std": 9.431926880652563,
		},
		"Appliance7":{
			"name": "Microwave",
			"thrd": 552,
			"mean": 1152.4315565031984,
			"std": 77.6351646676363,
		},
		"Appliance8":{
			"name": "Hi-Fi",
			"thrd": 1413,
			"mean": 2840.1637084819104,
			"std": 72.09542748747835,
		},
		"Appliance9":{
			"name": "Toaster",
			"thrd": 424,
			"mean": 964.6942328618063,
			"std": 101.63950555907408,
		},
	},
	# House 16
	16:{
		"Aggregate":{
			"thrd": 1581,
			"mean": 2765.2312929085733,
			"std": 1011.49430855692,
		},
		"Appliance1":{
			"name": "Fridge-Freezer(1)",
			"thrd": 50,
			"mean": 99.06835029328899,
			"std": 22.559709963847023,
		},
		"Appliance2":{
			"name": "Fridge-Freezer(2)",
			"thrd": 45,
			"mean": 90.68032063392889,
			"std": 56.95915144515666,
		},
		"Appliance3":{
			"name": "Electric_Heater(1)",
			"thrd": 352,
			"mean": 748.0204520990312,
			"std": 192.92101394440064,
		},
		"Appliance4":{
			"name": "Electric_Heater(2)",
			"thrd": 763,
			"mean": 1335.8223478939158,
			"std": 354.40942812802956,
		},
		"Appliance5":{
			"name": "Washing_Machine",
			"thrd": 951,
			"mean": 2010.7452252176117,
			"std": 54.224280196434044,
		},
		"Appliance6":{
			"name": "Dishwasher",
			"thrd": 1084,
			"mean": 2179.5550801363406,
			"std": 43.19980501495327,
		},
		"Appliance7":{
			"name": "Computer_Site",
			"thrd": 22,
			"mean": 40.743709418461606,
			"std": 12.581766867897448,
		},
		"Appliance8":{
			"name": "Television_Site",
			"thrd": 92,
			"mean": 181.75726884613053,
			"std": 35.06529684078795,
		},
		"Appliance9":{
			"name": "Dehumidifier",
			"thrd": 254,
			"mean": 508.248121400323,
			"std": 284.0770376883101,
		},
	},
	# House 17
	17:{
		"Aggregate":{
			"thrd": 1540,
			"mean": 2842.0064697311054,
			"std": 1712.7290225217496,
		},
		"Appliance1":{
			"name": "Freezer",
			"thrd": 72,
			"mean": 145.13513829825084,
			"std": 20.152110605421676,
		},
		"Appliance2":{
			"name": "Fridge-Freezer",
			"thrd": 40,
			"mean": 80.44192839945815,
			"std": 32.73514569056894,
		},
		"Appliance3":{
			"name": "Tumble_Dryer",
			"thrd": 1316,
			"mean": 2548.8827654409783,
			"std": 51.39974286372477,
		},
		"Appliance4":{
			"name": "Washing_Machine",
			"thrd": 996,
			"mean": 1991.2890199033645,
			"std": 171.91564291439468,
		},
		"Appliance5":{
			"name": "Computer_Site",
			"thrd": 35,
			"mean": 65.06186252975117,
			"std": 56.13992399892194,
		},
		"Appliance6":{
			"name": "Television_Site",
			"thrd": 38,
			"mean": 63.43367109932325,
			"std": 23.320086980357985,
		},
		"Appliance7":{
			"name": "Microwave",
			"thrd": 681,
			"mean": 1350.0142746684114,
			"std": 73.7677397635089,
		},
		"Appliance8":{
			"name": "Kettle",
			"thrd": 1461,
			"mean": 2942.240562808376,
			"std": 100.66391918151152,
		},
		"Appliance9":{
			"name": "TV_Site(Bedroom)",
			"thrd": 623,
			"mean": 1263.1396160558463,
			"std": 491.20416622910614,
		},
	},
	# House 18
	18:{
		"Aggregate":{
			"thrd": 1797,
			"mean": 3208.1732306141553,
			"std": 1155.8260763855578,
		},
		"Appliance1":{
			"name": "Fridge(garage)",
			"thrd": 58,
			"mean": 116.27572212791505,
			"std": 55.3261541262543,
		},
		"Appliance2":{
			"name": "Freezer(garage)",
			"thrd": 86,
			"mean": 172.76900656993027,
			"std": 49.09399416684078,
		},
		"Appliance3":{
			"name": "Fridge-Freezer",
			"thrd": 68,
			"mean": 128.45775461311914,
			"std": 34.86838903227868,
		},
		"Appliance4":{
			"name": "Washer_Dryer(garage)",
			"thrd": 896,
			"mean": 1777.3155302048224,
			"std": 536.6909193326487,
		},
		"Appliance5":{
			"name": "Washing_Machine",
			"thrd": 1128,
			"mean": 2153.4809439397295,
			"std": 99.32597399729107,
		},
		"Appliance6":{
			"name": "Dishwasher",
			"thrd": 1374,
			"mean": 2699.5219474234577,
			"std": 105.36087956479194,
		},
		"Appliance7":{
			"name": "Desktop_Computer",
			"thrd": 62,
			"mean": 105.8083948941502,
			"std": 31.950487217614295,
		},
		"Appliance8":{
			"name": "Television_Site",
			"thrd": 58,
			"mean": 95.05720195953504,
			"std": 14.805774195671043,
		},
		"Appliance9":{
			"name": "Microwave",
			"thrd": 802,
			"mean": 1605.0190302398332,
			"std": 290.0630526803696,
		},
	},
	# House 19
	19:{
		"Aggregate":{
			"thrd": 1367,
			"mean": 2496.0890386170254,
			"std": 699.5384119439444,
		},
		"Appliance1":{
			"name": "Fridge_Freezer",
			"thrd": 49,
			"mean": 98.49023886494886,
			"std": 42.16859845994051,
		},
		"Appliance2":{
			"name": "Washing_Machine",
			"thrd": 1128,
			"mean": 2248.4042269611577,
			"std": 165.84785874490152,
		},
		"Appliance3":{
			"name": "Television_Site",
			"thrd": 34,
			"mean": 61.138226680222765,
			"std": 24.46830128326777,
		},
		"Appliance4":{
			"name": "Microwave",
			"thrd": 633,
			"mean": 1271.6433397683397,
			"std": 56.861488804468,
		},
		"Appliance5":{
			"name": "Kettle",
			"thrd": 1464,
			"mean": 2951.2469396369775,
			"std": 88.29031390505176,
		},
		"Appliance6":{
			"name": "Toaster",
			"thrd": 445,
			"mean": 881.9889647039003,
			"std": 39.85053536479372,
		},
		"Appliance7":{
			"name": "Bread-maker",
			"thrd": 268,
			"mean": 537.4539905175819,
			"std": 66.85033272614746,
		},
		"Appliance8":{
			"name": "Games_Console",
			"thrd": 38,
			"mean": 77.70454227205346,
			"std": 30.522697027533987,
		},
		"Appliance9":{
			"name": "Hi-Fi",
			"thrd": 868,
			"mean": 1756.65,
			"std": 620.9696602792618,
		},
	},
	# House 20
	20:{
		"Aggregate":{
			"thrd": 1371,
			"mean": 2449.3128085871153,
			"std": 792.1849795283719,
		},
		"Appliance1":{
			"name": "Fridge",
			"thrd": 43,
			"mean": 86.70501081309268,
			"std": 23.236765857422267,
		},
		"Appliance2":{
			"name": "Freezer",
			"thrd": 60,
			"mean": 119.3090040107652,
			"std": 36.710353187139276,
		},
		"Appliance3":{
			"name": "Tumble_Dryer",
			"thrd": 768,
			"mean": 1535.7026491431109,
			"std": 42.03171447086585,
		},
		"Appliance4":{
			"name": "Washing_Machine",
			"thrd": 1132,
			"mean": 2231.9208213256484,
			"std": 70.34845695103323,
		},
		"Appliance5":{
			"name": "Dishwasher",
			"thrd": 1168,
			"mean": 2189.636788987278,
			"std": 73.80917353152982,
		},
		"Appliance6":{
			"name": "Computer_Site",
			"thrd": 52,
			"mean": 105.44070168242354,
			"std": 26.475497105084482,
		},
		"Appliance7":{
			"name": "Television_Site",
			"thrd": 39,
			"mean": 72.68861667830454,
			"std": 23.57298080828076,
		},
		"Appliance8":{
			"name": "Microwave",
			"thrd": 650,
			"mean": 1307.6321654501216,
			"std": 64.97752133003696,
		},
		"Appliance9":{
			"name": "Kettle",
			"thrd": 1381,
			"mean": 2781.7728615990804,
			"std": 92.74844861517893,
		},
	},
	# House 21
	21:{
		"Aggregate":{
			"thrd": 1053,
			"mean": 1822.4505474337566,
			"std": 576.7490113814079,
		},
		"Appliance1":{
			"name": "Fridge-Freezer",
			"thrd": 48,
			"mean": 95.58431859079593,
			"std": 24.28969543972803,
		},
		"Appliance2":{
			"name": "Tumble_Dryer",
			"thrd": 725,
			"mean": 1456.326051215329,
			"std": 217.44075623378458,
		},
		"Appliance3":{
			"name": "Washing_Machine",
			"thrd": 1051,
			"mean": 2087.851569338502,
			"std": 83.32596704160633,
		},
		"Appliance4":{
			"name": "Dishwasher",
			"thrd": 1037,
			"mean": 2066.0512368179957,
			"std": 57.25161675325738,
		},
		"Appliance5":{
			"name": "Food_Mixer",
			"thrd": 33,
			"mean": 66.4104104104104,
			"std": 36.06887073401058,
		},
		"Appliance6":{
			"name": "Television",
			"thrd": 17,
			"mean": 35.07546299823181,
			"std": 31.474916164756426,
		},
		"Appliance7":{
			"name": "Kettle",
			"thrd": 946,
			"mean": 1932.6096181780451,
			"std": 712.8757728134859,
		},
		"Appliance8":{
			"name": "Vivarium",
			"thrd": 15,
			"mean": 20.165192774659232,
			"std": 1.9969441498343639,
		},
		"Appliance9":{
			"name": "Pond_Pump",
			"thrd": 22,
			"mean": 44.46937721672841,
			"std": 7.69781032055738,
		},
	},

}
