import torch

if __name__ == "__main__":
        t = torch.randn(3,2,1)



        lines = t.movedim(3, -1)