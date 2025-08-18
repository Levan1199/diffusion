from .modules import *

class SwinTransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Stage1 = AlternatingEncoderBlock(24, 3)
        self.Stage2 = AlternatingEncoderBlock(48, 6)
        self.Stage3_1 = AlternatingEncoderBlock(96, 12)
        self.Stage3_2 = AlternatingEncoderBlock(96, 12)
        self.Stage3_3 = AlternatingEncoderBlock(96, 12)
        self.Stage4 = AlternatingEncoderBlock(192, 24)

        self.PatchExpanding4 = PatchExpand(192)
        self.PatchExpanding3 = PatchExpand(96)
        self.PatchExpanding2 = PatchExpand(48)

        self.output = SwinOutput()


    def forward(self, x):
        x = self.PatchExpanding4(self.Stage4(x))
        x = self.Stage3_1(x)
        x = self.Stage3_2(x)
        x = self.Stage3_3(x)
        x = self.PatchExpanding3(x)
        x = self.PatchExpanding2(self.Stage2(x))
        x = self.output(x)
        return x