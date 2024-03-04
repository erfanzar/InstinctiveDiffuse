from src.instinctive_diffuse import GradioUserInterface, SamplerConfig, TextToImageSampler


def main():
    sampler_config = SamplerConfig()
    sampler = TextToImageSampler.from_pretrained(
        "Instinct-AI/InstinctiveDiffuse",
        sampler_config=sampler_config,
    )
    GradioUserInterface(sampler=sampler).create().launch(share=False)


if __name__ == "__main__":
    main()
