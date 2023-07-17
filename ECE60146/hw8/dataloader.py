'''
Dataloader script. Borrow from DL Studio
'''


from dlstudio import DLStudio

dataroot = "/home/tam/git/ece60146/data/hw8_dataset/"
dls = DLStudio(
    dataroot=dataroot,
    path_saved_model="./saved_model",
    momentum=0.9,
    learning_rate= 1e-4,
    epochs=5,
    batch_size=1,
    classes=('negative','positive'),
    use_gpu=True,
)

number = 400
dataset_archive_train = f"sentiment_dataset_train_{number}.tar.gz"
dataset_archive_test =  f"sentiment_dataset_test_{number}.tar.gz"
text_cl = DLStudio.TextClassificationWithEmbeddings(dl_studio=dls)

train = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
    train_or_test='train',
    dl_studio=dls,
    dataset_file=dataset_archive_train,
    path_to_saved_embeddings=dataroot,
)
test = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
    train_or_test='test',
    dl_studio=dls,
    dataset_file=dataset_archive_test,
    path_to_saved_embeddings=dataroot,
)
text_cl.dataserver_train = train
text_cl.dataserver_test = test
text_cl.load_SentimentAnalysisDataset(train, test)

if __name__ == '__main__':
    print(f'Training sample size = {len(train):,}')
    print(f'Testing sample size = {len(test):,}')
    