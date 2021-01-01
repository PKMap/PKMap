# -*- coding: utf-8 -*-
# === pkmap_data.py ===



# artifical Dataset for illustration propuse
AD = {'0000': 0, 
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



