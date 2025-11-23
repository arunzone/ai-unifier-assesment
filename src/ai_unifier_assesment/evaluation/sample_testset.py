"""
Sample evaluation testset for Lord of the Rings: Fellowship of the Ring.
Contains 25 graded questions with ground truth answers and contexts.
"""

SAMPLE_QUESTIONS = [
    {
        "question": "Who is the bearer of the One Ring in the Fellowship?",
        "ground_truth_answer": "Frodo Baggins is the bearer of the One Ring.",
        "ground_truth_contexts": ["Frodo Baggins", "bearer of the Ring", "Ring-bearer"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "What is the name of Gandalf's sword?",
        "ground_truth_answer": "Gandalf's sword is called Glamdring.",
        "ground_truth_contexts": ["Glamdring", "Gandalf's sword", "Foe-hammer"],
        "source_metadata": {"topic": "weapons"},
    },
    {
        "question": "Where was the Council of Elrond held?",
        "ground_truth_answer": "The Council of Elrond was held in Rivendell.",
        "ground_truth_contexts": ["Rivendell", "Council of Elrond", "Imladris"],
        "source_metadata": {"topic": "locations"},
    },
    {
        "question": "Who created the One Ring?",
        "ground_truth_answer": "Sauron created the One Ring in the fires of Mount Doom.",
        "ground_truth_contexts": ["Sauron", "Dark Lord", "Mount Doom", "forged the Ring"],
        "source_metadata": {"topic": "history"},
    },
    {
        "question": "What is the name of Frodo's gardener and companion?",
        "ground_truth_answer": "Samwise Gamgee is Frodo's gardener and loyal companion.",
        "ground_truth_contexts": ["Samwise Gamgee", "Sam", "gardener"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "What creature attacked the Fellowship in Moria?",
        "ground_truth_answer": "A Balrog attacked the Fellowship in Moria.",
        "ground_truth_contexts": ["Balrog", "Durin's Bane", "Moria", "shadow and flame"],
        "source_metadata": {"topic": "creatures"},
    },
    {
        "question": "Who is the leader of the Fellowship?",
        "ground_truth_answer": "Gandalf initially leads the Fellowship, then Aragorn takes over.",
        "ground_truth_contexts": ["Gandalf", "Aragorn", "leader", "Fellowship"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "What is Aragorn's royal lineage?",
        "ground_truth_answer": "Aragorn is the heir of Isildur and rightful King of Gondor.",
        "ground_truth_contexts": ["Isildur", "heir", "King of Gondor", "Aragorn"],
        "source_metadata": {"topic": "history"},
    },
    {
        "question": "What gift did Galadriel give to Frodo?",
        "ground_truth_answer": "Galadriel gave Frodo a phial containing the light of Eärendil's star.",
        "ground_truth_contexts": ["phial", "light of Eärendil", "Galadriel", "star-glass"],
        "source_metadata": {"topic": "items"},
    },
    {
        "question": "Who is the Elf member of the Fellowship?",
        "ground_truth_answer": "Legolas of the Woodland Realm is the Elf member of the Fellowship.",
        "ground_truth_contexts": ["Legolas", "Elf", "Woodland Realm", "Mirkwood"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "What is the name of Bilbo's sword?",
        "ground_truth_answer": "Bilbo's sword is called Sting.",
        "ground_truth_contexts": ["Sting", "Bilbo's sword", "glows blue", "Orcs"],
        "source_metadata": {"topic": "weapons"},
    },
    {
        "question": "Where do the Hobbits live?",
        "ground_truth_answer": "The Hobbits live in the Shire.",
        "ground_truth_contexts": ["Shire", "Hobbits", "Hobbiton", "Bag End"],
        "source_metadata": {"topic": "locations"},
    },
    {
        "question": "Who is Gimli's father?",
        "ground_truth_answer": "Gimli's father is Glóin.",
        "ground_truth_contexts": ["Glóin", "Gimli's father", "dwarf"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "What happened to Gandalf in Moria?",
        "ground_truth_answer": "Gandalf fell into the abyss while fighting the Balrog on the Bridge of Khazad-dûm.",
        "ground_truth_contexts": ["fell", "Balrog", "Bridge of Khazad-dûm", "You shall not pass"],
        "source_metadata": {"topic": "events"},
    },
    {
        "question": "What are the names of Bilbo's parents?",
        "ground_truth_answer": "Bilbo's parents are Bungo Baggins and Belladonna Took.",
        "ground_truth_contexts": ["Bungo Baggins", "Belladonna Took", "parents"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "What is mithril?",
        "ground_truth_answer": "Mithril is a precious silver-like metal mined by the Dwarves, stronger than steel.",
        "ground_truth_contexts": ["mithril", "silver", "Dwarves", "stronger than steel"],
        "source_metadata": {"topic": "items"},
    },
    {
        "question": "Who is Boromir's father?",
        "ground_truth_answer": "Boromir's father is Denethor, the Steward of Gondor.",
        "ground_truth_contexts": ["Denethor", "Steward of Gondor", "Boromir's father"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "What is Lothlórien?",
        "ground_truth_answer": "Lothlórien is an Elven forest realm ruled by Galadriel and Celeborn.",
        "ground_truth_contexts": ["Lothlórien", "Elven", "Galadriel", "Celeborn", "Golden Wood"],
        "source_metadata": {"topic": "locations"},
    },
    {
        "question": "How many rings were given to the Elven-kings?",
        "ground_truth_answer": "Three rings were given to the Elven-kings under the sky.",
        "ground_truth_contexts": ["Three Rings", "Elven-kings", "Narya", "Nenya", "Vilya"],
        "source_metadata": {"topic": "history"},
    },
    {
        "question": "What is the name of the inn in Bree?",
        "ground_truth_answer": "The inn in Bree is called The Prancing Pony.",
        "ground_truth_contexts": ["Prancing Pony", "Bree", "inn", "Butterbur"],
        "source_metadata": {"topic": "locations"},
    },
    {
        "question": "Who stabbed Frodo on Weathertop?",
        "ground_truth_answer": "The Witch-king of Angmar stabbed Frodo with a Morgul blade on Weathertop.",
        "ground_truth_contexts": ["Witch-king", "Morgul blade", "Weathertop", "Nazgûl"],
        "source_metadata": {"topic": "events"},
    },
    {
        "question": "What is Aragorn's elvish name?",
        "ground_truth_answer": "Aragorn's elvish name is Elessar, meaning 'Elfstone'.",
        "ground_truth_contexts": ["Elessar", "Elfstone", "Aragorn", "Strider"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "Who is the Lady of Rivendell?",
        "ground_truth_answer": "Arwen Undómiel is the Lady of Rivendell, daughter of Elrond.",
        "ground_truth_contexts": ["Arwen", "Undómiel", "Elrond's daughter", "Evenstar"],
        "source_metadata": {"topic": "characters"},
    },
    {
        "question": "What is the Elvish word for 'friend' that opens the Doors of Durin?",
        "ground_truth_answer": "Mellon is the Elvish word for 'friend' that opens the Doors of Durin.",
        "ground_truth_contexts": ["Mellon", "friend", "Doors of Durin", "Moria"],
        "source_metadata": {"topic": "languages"},
    },
    {
        "question": "Who are the nine members of the Fellowship?",
        "ground_truth_answer": "The nine members are Frodo, Sam, Merry, Pippin, Gandalf, Aragorn, Legolas, Gimli, and Boromir.",
        "ground_truth_contexts": [
            "Frodo",
            "Sam",
            "Merry",
            "Pippin",
            "Gandalf",
            "Aragorn",
            "Legolas",
            "Gimli",
            "Boromir",
            "nine",
        ],
        "source_metadata": {"topic": "characters"},
    },
]
