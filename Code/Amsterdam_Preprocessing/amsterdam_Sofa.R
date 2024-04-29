library(ricu)

import_src('aumc',data_dir='/work/AmsterdamUMCdb-v1.0.2')

attach_src('aumc',data_dir='/work/AmsterdamUMCdb-v1.0.2')

sofa=load_concepts('sofa','aumc')
write.csv(sofa,'/work/wendland/Documents/Amsterdamdata/Sofascore.csv')

sepsis3=load_concepts('sep3','aumc')
write.csv(sofa,'/work/wendland/Documents/Amsterdamdata/Sep3.csv')

