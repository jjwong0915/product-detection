import dataloader
import model
import os
import trainer

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def main():
    target_model = model.ProductDetectionModel()
    trainer.train(
        model=target_model,
        data=dataloader.load_data("./data/train/train/train"),
        checkpoint_path="./checkpoint/fe-{epoch:02d}-{accuracy:.2f}.h5",
    )


if __name__ == "__main__":
    main()
