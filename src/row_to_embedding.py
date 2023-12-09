from tqdm import tqdm

class RowToEmbedding:
    def __init__(self, sfb, ee, dp):
        self.sfb = sfb
        self.ee = ee
        self.dp = dp

    def row_to_embedding(self, row, option):
        to_return = list()

        if option == "full":
            path = self.dp.get_sdp_with_dep(text=row['text'],
                                            source_i=self.sfb.return_idx(row)[0],
                                            target_i=self.sfb.return_idx(row)[3])
        elif option == "notfull":
            path = self.dp.get_sdp_with_dep(text=row['text'],
                                            source_i=self.sfb.return_idx(row)[1],
                                            target_i=self.sfb.return_idx(row)[2])

        word_embedding_dictionary = self.sfb.build_embedding(row)

        for i in range(len(path)):
            ent1 = word_embedding_dictionary[str(path[i][0])]
            ent2 = word_embedding_dictionary[str(path[i][2])]
            embedding = (ent1, path[i][1], ent2)
            
            to_return.append(embedding)

        return to_return
    
    def row_to_embedding_df(self, df, option):
        to_return = list()
        for i in tqdm(range(len(df))):
            row = df.iloc[i]
            try: 
                embedding = self.row_to_embedding(row, option)
                to_return.append(embedding)
            except:
                print("error at row {}".format(i))
                to_return.append(None)
            
        return to_return