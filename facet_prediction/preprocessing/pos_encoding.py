def pos_avg_ratio_caculating(sids, doc):
    num_sents = sum(len(section['sents']) for section in doc['sections'])
    num_sects = len(doc['sections'])

    sec_poss = []
    sec_pos_ratios = []

    ssids = []
    ssid_ratios = []

    for sid in sids:
        section_number, num_sents_in_sec, ssid = get_pos_by_sid(doc, sid)
        sec_poss.append(section_number)
        sec_pos_ratios.append(section_number/num_sects)
        if ssid != None: ssids.append(ssid)
        if ssid != None: ssid_ratios.append(ssid/num_sents_in_sec)

    try: sec_pos_avg = sum(sec_poss) / len(sec_poss)
    except: sec_pos_avg = None

    try: ssid_avg = sum(ssids) / len(ssids) if len(ssids)>0 else 0
    except: ssid_avg = None

    try: sid_avg = sum(sids) / len(sids) if len(sids)>0 else 0
    except: sid_avg = None

    try: sec_pos_ratio_avg = sum(sec_pos_ratios)/len(sec_pos_ratios) if len(sec_pos_ratios)>0 else 0
    except: sec_pos_ratio_avg = None

    try: ssid_ratio_avg  = sum(ssid_ratios)/len(ssid_ratios) if len(ssid_ratios)>0 else 0
    except: ssid_ratio_avg = None

    try: sid_ratio_avg = sid_avg/num_sents
    except: sid_ratio_avg = None

    return sec_poss, sec_pos_avg, sec_pos_ratio_avg, ssid_avg, ssid_ratio_avg, sid_avg, sid_ratio_avg

def get_pos_by_sid(doc, sid):
    for section in doc['sections']:
        for sent in section['sents']:
            if sent['sid'] == sid:
                sec_num = int(section['number'])
                num_sents_in_sec = len(section['sents'])
                ssid = sent['ssid']
                return sec_num, num_sents_in_sec, ssid
    print(f"ERROR when find {sid} in {doc['ID']}")


