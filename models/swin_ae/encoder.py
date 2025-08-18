from models.swin_ae.modules import *

class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = SwinEmbedding(C=24)
        self.PatchMerge1 = PatchMerging(24)
        self.PatchMerge2 = PatchMerging(48)
        self.PatchMerge3 = PatchMerging(96)
        self.Stage1 = AlternatingEncoderBlock(24, 3)
        self.Stage2 = AlternatingEncoderBlock(48, 6)
        self.Stage3_1 = AlternatingEncoderBlock(96, 12)
        self.Stage3_2 = AlternatingEncoderBlock(96, 12)
        self.Stage3_3 = AlternatingEncoderBlock(96, 12)
        self.Stage4 = AlternatingEncoderBlock(192, 24)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.PatchMerge1(self.Stage1(x))
        x = self.PatchMerge2(self.Stage2(x))
        x = self.Stage3_1(x)
        x = self.Stage3_2(x)
        x = self.Stage3_3(x)
        x = self.PatchMerge3(x)
        x = self.Stage4(x)
        return x