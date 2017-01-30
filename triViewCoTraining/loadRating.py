from collections import defaultdict
def load(filename,c1_user,c2_product,c3_rating,separator):
    userProfile = defaultdict(dict)
    itemProfile = defaultdict(dict)
    with open (filename) as f:
        for line in f:
            items = line.strip().split(separator)
            userProfile[items[c1_user]][items[c2_product]] = float(items[c3_rating])
            itemProfile[items[c2_product]][items[c1_user]] = float(items[c3_rating])
    return userProfile,itemProfile

def loadBrand(filename):
    brandProfile = {}
    with open(filename) as f:
        for line in f:
            try:
                items = line.strip().split('\t')
                if len(items)<3:
                    continue
                brandProfile[items[0]] = items[2]
            except IndexError:
                pass
    return brandProfile