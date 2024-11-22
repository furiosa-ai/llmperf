import random
from typing import Tuple, Callable


def randomly_sample_translation_prompt(
    prompt_tokens_mean: int = 550,
    prompt_tokens_stddev: int = 250,
    expect_output_tokens: int = 150,
    get_token_len=Callable,
) -> Tuple[str, int]:
    # XXX: use this overly simplified condition for now
    if prompt_tokens_mean < 400:
        prompt = get_prompt_100()
    else:
        prompt = get_prompt_1000(get_token_len)

    return [prompt, get_token_len(prompt)]


def get_prompt_100():
    """
    Example:

    Translate each of these English sentences to both Spanish and Italian: 'Exploring the depths of the ocean reveals mysterious and unknown creatures.', 'The light from distant stars takes millions of years to reach us.', 'What is the speed of light?', 'The impact of climate change on global weather patterns is profound.', 'How do astronauts train for space missions?'
    """
    selected_lang = random.sample(TARGET_LANG, 2)
    selected_sentences = random.sample(TARGET_SHORT_SENTENCES, 5)
    out = (
        f"Translate each of these English sentences to both {selected_lang[0]} and {selected_lang[1]}: "
        + ", ".join([f"'{sentence}'" for sentence in selected_sentences])
    )
    return out


def get_prompt_1000(get_token_len):
    """
    Example:

    Below is a list of sentences. I'd like you to translate a few of them into French please:

    1. The development of space elevators, theoretical structures designed to transport materials from Earth's surface into space, represents a bold vision for future space exploration.
    2. The concept of cryovolcanoes, volcanoes that erupt water, ammonia, or methane instead of lava, on icy moons and dwarf planets reshapes our understanding of geological activity.
    3. The study of Martian soil and its potential to grow plants challenges and expands our understanding of agriculture in extreme environments.
    4. Synthetic biology, combining biology and engineering, aims to design and construct new biological parts, devices, and systems not found in the natural world.
    5. The Fibonacci sequence, a series of numbers where each number is the sum of the two preceding ones, appears in nature in the arrangement of seeds in a sunflower and the spirals of a pinecone.
    6. The mysteries of dark matter and dark energy, making up most of the universe's mass and energy, challenge our understanding of the cosmos.
    7. The concept of zero gravity, or microgravity, experienced by astronauts in space, affects human physiology in unique ways, from fluid distribution to muscle atrophy.
    8. The study of cosmic rays, high-energy particles from outer space that strike the Earth's atmosphere, provides insights into the universe's most energetic events.
    9. Archaeological excavations in the city of Pompeii, buried under volcanic ash from Mount Vesuvius in 79 AD, offer a snapshot of Roman life frozen in time.
    10. Renewable energy technologies, such as wind turbines and solar panels, play a crucial role in reducing our reliance on fossil fuels and mitigating climate change.
    11. Quantum algorithms, like Shor's algorithm for factoring large numbers, could revolutionize fields by making certain classically intractable problems solvable.
    12. The concept of photosynthesis in space, exploring how plants might grow in microgravity, could be key to sustaining long-term space travel.
    13. Graphene, a single layer of carbon atoms arranged in a two-dimensional lattice, exhibits remarkable properties, including exceptional strength and electrical conductivity.
    14. Nanotechnology, manipulating matter at the atomic or molecular scale, holds the potential to revolutionize industries from medicine to manufacturing.
    15. The genetic code, a complex language of life, carries instructions for the development, functioning, and reproduction of all living organisms.
    16. The phenomenon of solar flares, intense bursts of radiation from the sun's surface, can disrupt Earth's magnetic field and communications.
    17. The concept of space-time fabric, a model in physics that combines space and time into a single continuum, challenges our perceptions of the universe.
    18. The Stegosaurus, known for its distinctive row of kite-shaped plates along its back, used these plates for temperature regulation and, possibly, as a display to deter predators or attract mates.
    19. The concept of digital currency, such as Bitcoin, challenges traditional notions of money and financial transactions in the digital age.
    20. The discovery of the Higgs boson particle at the Large Hadron Collider provided key insights into the fundamental structure of the universe.
    21. Puppets, with their diverse forms and mechanisms, have been used for centuries in storytelling, reflecting cultural histories and human creativity.
    22. The Voyager spacecraft, humanity's most distant messengers, carry golden records with sounds and images intended to portray the diversity of life on Earth.
    23. The study of exobiology, or astrobiology, explores the potential for life beyond Earth, examining the conditions and chemistry that support life.
    24. The concept of wormholes, bridges through spacetime theorized to connect distant points in the universe, fascinates scientists and science fiction fans alike.
    25. The observation of gravitational waves, ripples in the fabric of spacetime caused by violent cosmic events, opens a new window for understanding the universe.
    26. The use of drones for environmental monitoring allows scientists to collect data from inaccessible areas, improving our understanding of natural processes.

    Please translate the following numbered sentences into French: 1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 24, 25, 26. Don't repeat the sentences in English. Only translate those 20 sentences. Number them in your response. Thank you!
    """

    selected_lang = random.choice(TARGET_LANG)

    max_num_input_sentences = 30
    max_num_output_sentences = 20

    # render target sentences
    selected_sentences = random.sample(TARGET_LONG_SENTENCES, max_num_input_sentences)
    max_target_sentence_prompt_tokens = 800  # magic number

    target_sentence_prompt = ""
    num_actual_input_sentences = 0
    while num_actual_input_sentences < max_num_input_sentences:
        this = f"{num_actual_input_sentences + 1}. {selected_sentences[num_actual_input_sentences]}\n"
        if (
            get_token_len(target_sentence_prompt + this)
            > max_target_sentence_prompt_tokens
        ):
            break
        target_sentence_prompt += this
        num_actual_input_sentences += 1

    num_output_sentences = min(max_num_output_sentences, num_actual_input_sentences)
    selected_outout_sentence_indices = sorted(
        random.sample(range(1, num_actual_input_sentences + 1), num_output_sentences)
    )

    prompt = f"Below is a list of sentences. I'd like you to translate a few of them into {selected_lang} please:\n\n"
    prompt += target_sentence_prompt
    prompt += (
        f"\nPlease translate the following numbered sentences into {selected_lang}: "
        + ", ".join([str(i) for i in selected_outout_sentence_indices])
        + f". Don't repeat the sentences in English. Only translate those {num_output_sentences} sentences. Number them in your response. Thank you!"
    )
    return prompt


TARGET_LANG = ["German", "Italian", "French", "Spanish"]


TARGET_SHORT_SENTENCES = """
Can humans live on other planets?
Exploring the depths of the ocean reveals mysterious and unknown creatures.
How do astronauts train for space missions?
How do rockets escape Earth's gravity?
How do telescopes help us see distant stars?
I think visiting Mars would be fun.
I want to see the rings of Saturn.
I want to walk on the moon one day.
Protecting endangered species is crucial for maintaining biodiversity.
Quantum computers could revolutionize the field of information technology.
Renewable energy sources are essential for sustainable living on Earth.
The Aurora Australis lights up the southern skies with vibrant colors.
The art of coding allows us to create complex digital worlds.
The beauty of the Northern Lights fascinates observers around the world.
The challenge of deep space exploration pushes the limits of human ingenuity.
The colonization of Mars represents humanity's next giant leap.
The concept of artificial intelligence challenges our understanding of consciousness.
The construction of the pyramids remains an engineering marvel.
The creation of the internet has connected the world like never before.
The development of vaccines has saved millions of lives worldwide.
The exploration of space continues to expand our knowledge of the universe.
The impact of climate change on global weather patterns is profound.
The invention of the wheel marked a significant leap in human history.
The light from distant stars takes millions of years to reach us.
The mysteries of the Bermuda Triangle have puzzled scientists for decades.
The phenomenon of global warming requires immediate action.
The preservation of historical monuments helps us remember our past.
The process of photosynthesis is vital for life on Earth.
The rise of virtual reality technology opens new doors for digital experiences.
The study of ancient civilizations uncovers the roots of modern culture.
The study of genetics provides insights into the complexity of life.
The universe is vast and mysterious.
Understanding the human genome opens new pathways in medical science.
What are the challenges of space travel?
What does zero gravity feel like?
What is the biggest planet in the solar system?
What is the biggest star in the universe?
What is the most common element in the Earth's crust?
What is the most common element in the universe?
What is the smallest planet in the solar system?
What is the speed of light?
Where is the International Space Station?
Will I get to visit Jupiter one day?
""".strip().split(
    "\n"
)

TARGET_LONG_SENTENCES = """
Archaeological excavations in the city of Pompeii, buried under volcanic ash from Mount Vesuvius in 79 AD, offer a snapshot of Roman life frozen in time.
Artificial intelligence systems, learning from vast datasets, are beginning to outperform humans in tasks ranging from image recognition to strategic game playing.
Black holes, the mysterious cosmic phenomena where gravity is so strong that not even light can escape, serve as gateways to understanding the limits of our physical laws.
CRISPR-Cas9, a revolutionary gene-editing technology, allows for precise modifications to DNA, offering potential cures for genetic diseases.
Cognitive behavioral therapy, a form of psychological treatment, has been effective in treating a range of mental health conditions by changing patterns of thinking and behavior.
Coral reefs, often referred to as the rainforests of the sea, are diverse underwater ecosystems vital for marine life but threatened by climate change and pollution.
Dinosaurs, the dominant terrestrial vertebrates for over 160 million years, continue to fascinate us with their diversity and the mystery of their extinction.
Graphene, a single layer of carbon atoms arranged in a two-dimensional lattice, exhibits remarkable properties, including exceptional strength and electrical conductivity.
In the vast expanse of the universe, countless galaxies each harbor millions of stars, with many potentially hosting Earth-like planets.
Mathematics, the language of the universe, helps us understand patterns and structures in nature, from the Fibonacci sequence in flowers to the fractal patterns of snowflakes.
Nanotechnology, manipulating matter at the atomic or molecular scale, holds the potential to revolutionize industries from medicine to manufacturing.
Ocean acidification, caused by the absorption of CO2 from the atmosphere, threatens marine ecosystems and the biodiversity they support.
Photosynthesis, a process used by plants and some microorganisms, converts light energy into chemical energy, producing oxygen as a byproduct and sustaining life on Earth.
Puppets, with their diverse forms and mechanisms, have been used for centuries in storytelling, reflecting cultural histories and human creativity.
Quantum algorithms, like Shor's algorithm for factoring large numbers, could revolutionize fields by making certain classically intractable problems solvable.
Quantum computers operate using qubits, which unlike classical bits, can exist in multiple states at once, enabling unprecedented computational power.
Quantum entanglement, a phenomenon where particles become interconnected and share states instantaneously over distance, challenges our classical understanding of physics.
Quantum entanglement, a phenomenon where qubits become interconnected and the state of one can instantly influence another, is key to quantum computing's speed and efficiency.
Recent discoveries of well-preserved dinosaur fossils with soft tissues and proteins have opened new avenues for understanding dinosaur biology and their environment.
Renewable energy technologies, such as wind turbines and solar panels, play a crucial role in reducing our reliance on fossil fuels and mitigating climate change.
Space exploration has allowed humanity to set foot on the Moon, send probes to distant planets, and peer into the depths of the cosmos with powerful telescopes.
Synthetic biology, combining biology and engineering, aims to design and construct new biological parts, devices, and systems not found in the natural world.
The Aurora Borealis, or Northern Lights, paints the sky with breathtaking colors, a result of solar winds interacting with the Earth's magnetic field and atmosphere.
The Cassini spacecraft's mission to Saturn involved detailed studies of the planet's rings and moons, including Titan and Enceladus, revealing complex organic chemistry and potential habitats for life.
The Fibonacci sequence, a series of numbers where each number is the sum of the two preceding ones, appears in nature in the arrangement of seeds in a sunflower and the spirals of a pinecone.
The Great Barrier Reef, the world's largest coral reef system, is home to a vast array of marine species and is visible from space.
The Internet of Things (IoT) is transforming everyday objects into smart devices, creating a network of interconnected devices that can collect and exchange data.
The Juno probe, orbiting Jupiter, is studying the planet's composition, gravity field, magnetic field, and polar magnetosphere, offering insights into the gas giant's formation and structure.
The Large Hadron Collider, the world's largest and most powerful particle accelerator, has been instrumental in advancing our understanding of the fundamental particles of the universe.
The Mars Rover Perseverance, launched by NASA, is designed to explore the Jezero crater on Mars, searching for signs of ancient life and collecting samples for future return to Earth.
The New Horizons mission, which flew by Pluto in 2015, transformed our understanding of this distant world, revealing mountains, glaciers, and an atmosphere more complex than previously thought.
The Stegosaurus, known for its distinctive row of kite-shaped plates along its back, used these plates for temperature regulation and, possibly, as a display to deter predators or attract mates.
The Tyrannosaurus Rex, one of the largest land predators to have ever lived, had a bite force of around 8,000 pounds per square inch, rivaling that of modern predators.
The Voyager 1 probe, now in interstellar space, has provided humanity with invaluable data about the outer planets and beyond, including the famous Pale Blue Dot image.
The Voyager spacecraft, humanity's most distant messengers, carry golden records with sounds and images intended to portray the diversity of life on Earth.
The advancements in genetic engineering, including CRISPR technology, open new possibilities for medicine, agriculture, and environmental protection.
The biodiversity of the Amazon rainforest, a treasure trove of species, plays a crucial role in the global ecosystem and climate regulation.
The concept of cryovolcanoes, volcanoes that erupt water, ammonia, or methane instead of lava, on icy moons and dwarf planets reshapes our understanding of geological activity.
The concept of digital currency, such as Bitcoin, challenges traditional notions of money and financial transactions in the digital age.
The concept of parallel universes, a staple of science fiction, is explored in physics theories, suggesting that our universe might be one of many in a vast multiverse.
The concept of photosynthesis in space, exploring how plants might grow in microgravity, could be key to sustaining long-term space travel.
The concept of planetary habitability, assessing the potential of other worlds to support life, guides the search for extraterrestrial life in the universe.
The concept of space-time fabric, a model in physics that combines space and time into a single continuum, challenges our perceptions of the universe.
The concept of sustainable development seeks to meet the needs of the present without compromising the ability of future generations to meet their own needs.
The concept of the Anthropocene epoch suggests that human activity has become the dominant influence on climate and the environment.
The concept of time dilation in special relativity shows that time passes at different rates for observers in different frames of reference.
The concept of wormholes, bridges through spacetime theorized to connect distant points in the universe, fascinates scientists and science fiction fans alike.
The concept of zero gravity, or microgravity, experienced by astronauts in space, affects human physiology in unique ways, from fluid distribution to muscle atrophy.
The creation of artificial reefs, using sunken ships or other structures, promotes marine biodiversity and supports conservation efforts.
The dance of the Northern and Southern Lights, caused by solar particles colliding with the Earth's atmosphere, showcases nature's grand spectacle.
The delicate balance of ecosystems, where each species plays a specific role, highlights the importance of conservation efforts to protect biodiversity.
The development of autonomous robots capable of navigating complex environments opens new possibilities for exploration and service in areas inaccessible to humans.
The development of quantum computing, based on the principles of quantum mechanics, promises to revolutionize information processing and technology.
The development of quantum error correction codes is crucial for stabilizing qubits against decoherence, ensuring the reliability of quantum computations.
The development of smart cities, using technology to improve the efficiency of services and meet residents' needs, represents a new frontier in urban planning.
The development of space elevators, theoretical structures designed to transport materials from Earth's surface into space, represents a bold vision for future space exploration.
The discovery of bioluminescent waves, where certain conditions cause ocean waters to glow at night, offers a glimpse into the unseen beauty of the marine world.
The discovery of feathered dinosaurs in the late 20th century has reshaped our understanding of the evolution of birds and their connection to these ancient creatures.
The discovery of penicillin by Alexander Fleming in 1928, a breakthrough in medical science, has saved countless lives by treating bacterial infections effectively.
The discovery of the Higgs boson particle at the Large Hadron Collider provided key insights into the fundamental structure of the universe.
The discovery of the Terracotta Army, a collection of sculptures depicting the armies of Qin Shi Huang, the first Emperor of China, provides insight into ancient military and cultural practices.
The discovery of water in the form of ice on the Moon opens new possibilities for future lunar exploration and the potential for human colonization.
The exploration of Mars, our neighboring planet, through rovers and satellites, seeks to uncover its secrets and the possibility of past or present life.
The exploration of the deep sea reveals unique life forms adapted to extreme conditions, expanding our knowledge of biological diversity.
The exploration of the human microbiome, the collection of trillions of microbes living in and on our bodies, is revealing new insights into health and disease.
The exploration of underwater caves, revealing hidden ecosystems and geological formations, mirrors the exploration of space in its complexity and unknowns.
The fascinating world of quantum mechanics reveals that particles can exist in multiple states simultaneously until observed.
The formation of diamonds in the intense pressure and heat of the Earth's mantle, then brought to the surface by volcanic eruptions, mirrors the planet's dynamic processes.
The genetic code, a complex language of life, carries instructions for the development, functioning, and reproduction of all living organisms.
The global decline in bee populations poses a significant threat to pollination, affecting food production and ecosystem health.
The human brain, a complex organ with billions of neurons, is capable of extraordinary feats of creativity, problem-solving, and emotional depth.
The impact of comets and asteroids on Earth's history, including theories about their role in mass extinctions, adds a dynamic element to our planet's story.
The intricate dance of planets around stars in distant solar systems, known as exoplanets, expands our horizons in the search for extraterrestrial life.
The intricate mechanisms of photosynthesis, converting sunlight into energy, underscore the interconnectedness of life and the environment.
The lifecycle of a star, from its birth in a nebula to its death as a supernova or a black hole, tells the story of the universe's evolution and the elements that make up our world.
The lifecycle of a star, from its formation in a stellar nursery to its final stages as a supernova or a white dwarf, mirrors the cycle of birth, life, and death.
The mass extinction event that wiped out the dinosaurs 65 million years ago is believed to have been caused by a combination of volcanic activity and a catastrophic asteroid impact.
The melting of polar ice caps, accelerated by global warming, contributes to rising sea levels and the loss of habitat for species like the polar bear.
The mysteries of dark matter and dark energy, making up most of the universe's mass and energy, challenge our understanding of the cosmos.
The mystery of the Tunguska event, a massive explosion in Siberia in 1908, thought to be caused by a comet or asteroid, remains one of the 20th century's great enigmas.
The observation of gravitational waves, ripples in the fabric of spacetime caused by violent cosmic events, opens a new window for understanding the universe.
The phenomenon of bioluminescence, where living organisms produce light, creates magical scenes in the depths of the ocean and in the night landscapes.
The phenomenon of rain on different planets, such as methane rain on Titan, Saturn's largest moon, highlights the diversity of weather in our solar system.
The phenomenon of social media has transformed communication, creating new opportunities and challenges for privacy, information sharing, and digital literacy.
The phenomenon of solar flares, intense bursts of radiation from the sun's surface, can disrupt Earth's magnetic field and communications.
The phenomenon of superconductivity, where materials conduct electricity without resistance at low temperatures, holds promise for future technologies.
The phenomenon of the greenhouse effect, warming the Earth's surface through the trapping of solar heat by atmospheric gases, underscores the delicate balance of our climate system.
The physics of blackbody radiation, explaining how objects emit radiation based on their temperature, has applications in astronomy and thermal imaging.
The principle of superposition in quantum computing allows qubits to perform multiple calculations simultaneously, drastically reducing processing time for complex problems.
The process of metamorphosis in butterflies, from a crawling caterpillar to a beautiful winged creature, symbolizes transformation and the incredible adaptability of nature.
The process of nuclear fusion, which powers the sun, transforms hydrogen into helium, releasing vast amounts of energy and light.
The rings of Saturn, made up of billions of ice particles, orbit the planet in a stunning display of cosmic beauty and complexity.
The role of gravity in shaping the cosmos, from the formation of stars and planets to the dynamics of galaxies, highlights its fundamental force in the universe.
The search for exomoons, moons orbiting planets outside our solar system, could reveal new habitats for life.
The study of Martian soil and its potential to grow plants challenges and expands our understanding of agriculture in extreme environments.
The study of ancient DNA, extracted from fossils, is providing new insights into the evolution of species and migration patterns of human populations.
The study of cosmic rays, high-energy particles from outer space that strike the Earth's atmosphere, provides insights into the universe's most energetic events.
The study of exobiology, or astrobiology, explores the potential for life beyond Earth, examining the conditions and chemistry that support life.
The study of exoplanets, planets orbiting stars outside our solar system, has revealed a wide variety of worlds, from gas giants larger than Jupiter to rocky planets that may harbor liquid water.
The study of extremophiles, organisms that thrive in Earth's most inhospitable environments, offers clues about life's potential to exist on other planets.
The study of the human brain, through techniques like functional magnetic resonance imaging (fMRI), reveals the complex neural networks involved in perception, thought, and emotion.
The theory of plate tectonics explains the movement of the Earth's lithosphere, leading to the formation of mountains, earthquakes, and volcanic activity.
The theory of relativity, proposed by Albert Einstein, revolutionized our understanding of space, time, and gravity, challenging the notions of absolute space and time.
The transition to electric vehicles (EVs) is seen as a critical step in reducing greenhouse gas emissions and combating climate change.
The use of drones for environmental monitoring allows scientists to collect data from inaccessible areas, improving our understanding of natural processes.
The use of satellite technology to monitor climate change, track natural disasters, and explore remote areas of Earth demonstrates the intersection of technology and environmental science.
The water cycle, an essential process for life on Earth, involves the continuous movement of water through evaporation, condensation, precipitation, and runoff.
Urban farming, the practice of cultivating, processing, and distributing food in or around urban areas, offers a sustainable way to meet city dwellers' food needs.
Virtual reality technology immerses users in a computer-generated environment, opening new possibilities for gaming, education, and therapy.
""".strip().split(
    "\n"
)
