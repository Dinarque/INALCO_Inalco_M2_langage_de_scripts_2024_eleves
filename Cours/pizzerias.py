import asyncio
import time


# La pizzeria séquentielle

def servir_client_seq(client):
    # Étape 1 : Le serveur prend la commande
    print(f"👨‍🍳 Serveur : \"{client}, que désirez-vous ?\"")
    time.sleep(3)
    print(f"📋 Serveur : \"Commande de {client} transmise au cuisinier.\"")

    # Étape 2 : Le cuisinier prépare la pizza
    print(f"🍅 Cuisinier : \"Je prépare la pizza de {client}...\"")
    time.sleep(5)
    print(f"🔥 Cuisinier : \"Pizza de {client} au four !\"")

    # Étape 3 : La pizza est au four
    time.sleep(10)

    # Étape 4 : La pizza est prête
    print(f"🍕 {client} : \"Mmmh, ma pizza est là, merci !\"\n")

def pizzeria_seq():
    start = time.time()
    print("🏫 La classe arrive à la Pizzeria Asynchrone !\n")

    # Liste des clients
    clients = [f"Client {i + 1}" for i in range(20)]

    # Servir chaque client séquentiellement
    for client in clients:
        servir_client_seq(client)

    end = time.time()

    print(f"\n⏱️ Temps total pour servir toute la classe : {round(end - start, 2)} secondes.\n")
    print("🎉 Tout le monde est repu et prêt à repartir heureux ! 🍕")




# La pizzeria asynchrone facile

# Locks pour le serveur et le cuisinier
serveur_lock = asyncio.Lock()
cuisinier_lock = asyncio.Lock()

async def servir_client_async(client):
    async with serveur_lock:
        print(f"👨‍🍳 Serveur : \"{client}, que désirez-vous ?\"")
        await asyncio.sleep(3)
        print(f"📋 Serveur : \"Commande de {client} transmise au cuisinier.\"")

    async with cuisinier_lock:
        print(f"🍅 Cuisinier : \"Je prépare la pizza de {client}...\"")
        await asyncio.sleep(5)
        print(f"🔥 Cuisinier : \"Pizza de {client} au four !\"")

    await asyncio.sleep(10)
    print(f"🍕 {client} : \"Mmmh, ma pizza est là, merci !\"\n")

async def pizzeria_async():
    start = time.time()
    print("🏫 La classe arrive à la Pizzeria Asynchrone !\n")
    clients = [f"Client {i + 1}" for i in range(20)]
    tasks = [servir_client_async(client) for client in clients]
    await asyncio.gather(*tasks)

    end = time.time()

    print(f"\n⏱️ Temps total pour servir toute la classe : {round(end - start, 2)} secondes.\n")
    print("🎉 Tout le monde est repu et prêt à repartir heureux ! 🍕")



# La pizzeria asynchrone épicée

# Locks pour le serveur, le cuisinier et le four
four_lock = asyncio.Lock()
four_semaphore = asyncio.Semaphore(4)  # Le four peut accueillir 4 pizzas à la fois


async def servir_client_spicy(client, four_emplacements):
    async with serveur_lock:
        print(f"👨‍🍳 Serveur : \"{client}, que désirez-vous ?\"")
        await asyncio.sleep(3)
        print(f"📋 Serveur : \"Commande de {client} transmise au cuisinier.\"")

    async with cuisinier_lock:
        print(f"🍅 Cuisinier : \"Je prépare la pizza de {client}...\"")
        await asyncio.sleep(5)

    async with four_semaphore:
        async with four_lock: # Trouver un emplacement libre dans le four
            emplacement = None
            for i in range(len(four_emplacements)):
                if not four_emplacements[i]:
                    four_emplacements[i] = True
                    emplacement = i + 1  # Emplacements numérotés à partir de 1
                    break

        print(f"🔥 Cuisinier : \"Pizza de {client} mise au four à l'emplacement {emplacement} !\"")
        await asyncio.sleep(10)
        print(f"🔥 Cuisinier : \"Pizza de {client} cuite, on libère l'emplacement {emplacement} !\"")

        async with four_lock: # Libérer l'emplacement après cuisson
            four_emplacements[emplacement - 1] = False

    async with cuisinier_lock:
        print(f"🍕 Cuisinier : \"Je découpe la pizza de {client} et ajoute de l'huile piquante...\"")
        await asyncio.sleep(2)

    async with serveur_lock:
        print(f"🚚 Serveur : \"J'apporte la pizza au {client}.\"")
        await asyncio.sleep(3)
    print(f"🍕 {client} : \"Mmmh, ma pizza est là, merci !\"\n")


async def pizzeria_spicy():
    # Suivi des emplacements du four
    four_emplacements = [False, False, False, False]  # False = libre, True = occupé

    start = time.time()
    print("🏫 La classe arrive à la Pizzeria Asynchrone Epicée !\n")
    clients = [f"Client {i + 1}" for i in range(20)]
    tasks = [servir_client_spicy(client, four_emplacements) for client in clients]
    await asyncio.gather(*tasks)

    end = time.time()
    print(f"\n⏱️ Temps total pour servir toute la classe : {round(end - start, 2)} secondes.\n")
    print("🎉 Tout le monde est repu et prêt à repartir heureux ! 🍕")




if __name__ == "__main__" :
    pizzeria_seq()
    #asyncio.run(pizzeria_async())
    #asyncio.run(pizzeria_spicy())