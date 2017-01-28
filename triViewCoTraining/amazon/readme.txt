Dataset file specification:

1) reviews_mpro.txt (Character sets: GB2312): 
A snapshot of manufacturing product reviews from amazon.cn till August 20, 2012. Each line corresponds to a user-generated review with 10 attributes shown as below:
reviewer id, product id, username, rating, timestamp, review title, review content, number of helpfulness votes, number of overall votes, product category. (separated by Tab)

2) candidate_groups.txt
a set of candidate groups with each consisting of at least 2 reviewers (considered as members) who have reviewed at least 3 products together. Each line corresponds to a candidate group with the following format:
number of members, number of products reviewed by members, member ids, product ids. (separated by Tab)

3) colluder_labels.txt
Annotations for colluders and non-colluders. Each line corresponds to the annotation for a reviewer with the following format:
reviewer id, label. (separated by Tab, 0 for non-colluders, 1 for colluders)

4) product_brand.txt
Brand information for products. Each line corresponds to the brand of a product with the following format:
product id, product name, brand name. (separated by Tab)

-------------------------

We encourage you to cite our dataset if you have used them in your work. You can use the following BibTeX citation:
@inproceedings{xu2013uncovering,
            title={Uncovering collusive spammers in Chinese review Websites}, 
            author={Xu, Chang and Zhang, Jie and Chang, Kuiyu and Long, Chong}, 
            booktitle={Proceedings of the 22nd ACM International Conference on Information and Knowledge Management}, 
            pages={979--988},
            year={2013}
}