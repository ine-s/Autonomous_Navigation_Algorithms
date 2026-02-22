import heapq
import random
import math
import os

# Pour la visualisation graphique (Question 2)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import ListedColormap
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
    OUTPUT_DIR = "output_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib non disponible. Installer avec 'pip install matplotlib numpy' pour la visualisation graphique.")


class Maze:
    def __init__(self, filename):
        """
        Charge le labyrinthe depuis un fichier texte.
        - filename: chemin vers le fichier du labyrinthe
        
        Symboles:
        - '#' = mur (non traversable)
        - '.' = case libre
        - 'A' = départ
        - 'B' = arrivée
        """
        self.grid = []  # Le labyrinthe sous forme de liste 2D
        self.start = None  # Coordonnées du point de départ (i, j)
        self.goal = None   # Coordonnées du point d'arrivée (i, j)
        
        with open(filename, 'r') as f:
            for line in f:
                # On enlève le '\n' à la fin et on convertit en liste de caractères
                stripped = line.strip()
                if stripped:  # Ignorer les lignes vides
                    self.grid.append(list(stripped))
        
        self.height = len(self.grid)      # Nombre de lignes
        self.width = len(self.grid[0]) if self.grid else 0    # Nombre de colonnes
        
        # Matrice de récompense (coût pour traverser chaque case)
        # Par défaut: 1 pour case libre, infini pour mur
        self.rewards = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Parcourir la grille pour trouver A, B et initialiser les récompenses
        for i in range(self.height):
            for j in range(len(self.grid[i])):  # Utiliser la longueur réelle de chaque ligne
                cell = self.grid[i][j]
                if cell == 'A':
                    self.start = (i, j)
                    self.rewards[i][j] = 1  # Coût de traversée
                elif cell == 'B':
                    self.goal = (i, j)
                    self.rewards[i][j] = 1  # Coût de traversée
                elif cell == '#':
                    self.rewards[i][j] = float('inf')  # Mur = coût infini
                else:
                    self.rewards[i][j] = 1  # Case libre = coût 1
    
    def in_bounds(self, i, j):
        """
        Vérifie si la cellule (i, j) appartient à la grille.
        - i: numéro de ligne
        - j: numéro de colonne
        Retourne True si la cellule est dans les limites de la grille.
        """
        return 0 <= i < self.height and 0 <= j < self.width
    
    def is_valid(self, i, j):
        """
        Vérifie si la position (i, j) est valide et traversable.
        - i: numéro de ligne
        - j: numéro de colonne
        Retourne True si on peut marcher sur cette case.
        """
        # Vérifier que c'est dans les limites
        if not self.in_bounds(i, j):
            return False
        # Vérifier que ce n'est pas un mur
        return self.grid[i][j] != '#'
    
    def get_neighbors(self, i, j, allow_diagonal=False):
        """
        Retourne la liste des voisins valides de la case (i, j) avec leur coût.
        
        Paramètres:
        - i, j: position actuelle
        - allow_diagonal: si True, autorise les déplacements diagonaux
        
        Retourne: liste de tuples ((ni, nj), cout)
        """
        neighbors = []
        
        # Les 4 directions cardinales: (delta_i, delta_j, coût)
        directions = [
            (-1, 0, 1.0),   # haut
            (1, 0, 1.0),    # bas
            (0, -1, 1.0),   # gauche
            (0, 1, 1.0)     # droite
        ]
        
        # Ajouter les 4 directions diagonales si autorisé
        if allow_diagonal:
            diag_cost = math.sqrt(2)  # ≈ 1.414
            directions.extend([
                (-1, -1, diag_cost),  # haut-gauche
                (-1, 1, diag_cost),   # haut-droite
                (1, -1, diag_cost),   # bas-gauche
                (1, 1, diag_cost)     # bas-droite
            ])
        
        for di, dj, cost in directions:
            ni, nj = i + di, j + dj
            if self.is_valid(ni, nj):
                neighbors.append(((ni, nj), cost))
        
        return neighbors
    
    # ==================== PARTIE B : Fonctions de génération ====================
    
    @classmethod
    def create_empty(cls, height, width, start=(1, 1), goal=None):
        """
        Crée un labyrinthe vide (sans charger de fichier).
        
        Paramètres:
        - height: nombre de lignes
        - width: nombre de colonnes
        - start: tuple (i, j) pour le départ
        - goal: tuple (i, j) pour l'arrivée (par défaut: coin opposé)
        
        Retourne: une instance de Maze
        """
        maze = cls.__new__(cls)
        maze.height = height
        maze.width = width
        maze.start = start
        maze.goal = goal if goal else (height - 2, width - 2)
        
        # Créer une grille vide avec murs sur les bords
        maze.grid = []
        for i in range(height):
            row = []
            for j in range(width):
                if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                    row.append('#')  # Mur sur les bords
                else:
                    row.append('.')  # Case libre
            maze.grid.append(row)
        
        # Placer départ et arrivée
        maze.grid[maze.start[0]][maze.start[1]] = 'A'
        maze.grid[maze.goal[0]][maze.goal[1]] = 'B'
        
        # Initialiser la matrice de récompense
        maze.rewards = [[0 for _ in range(width)] for _ in range(height)]
        maze.init_rewards()
        
        return maze
    
    def generate_obstacles_random(self, density=0.2, seed=None):
        """
        Génère des obstacles de manière aléatoire.
        
        Paramètres:
        - density: proportion de cases qui deviennent des murs (0.0 à 1.0)
        - seed: graine pour reproductibilité (optionnel)
        
        Les cellules de départ et d'arrivée restent toujours franchissables.
        """
        if seed is not None:
            random.seed(seed)
        
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # Ne pas toucher au départ et à l'arrivée
                if (i, j) == self.start or (i, j) == self.goal:
                    continue
                
                # Placer un mur avec probabilité = density
                if random.random() < density:
                    self.grid[i][j] = '#'
                else:
                    self.grid[i][j] = '.'
        
        # Mettre à jour les récompenses
        self.init_rewards()
    
    def generate_obstacles_deterministic(self, pattern='grid', spacing=4):
        """
        Génère des obstacles de manière déterministe selon un motif.
        
        Paramètres:
        - pattern: 'grid' (grille), 'horizontal' (lignes), 'vertical' (colonnes)
        - spacing: espacement entre les obstacles
        
        Les cellules de départ et d'arrivée restent toujours franchissables.
        """
        # D'abord, nettoyer la grille (garder seulement les bords)
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                self.grid[i][j] = '.'
        
        for i in range(1, self.height - 1):
            for j in range(1, self.width - 1):
                # Ne pas toucher au départ et à l'arrivée
                if (i, j) == self.start or (i, j) == self.goal:
                    continue
                
                place_wall = False
                
                if pattern == 'grid':
                    # Grille: murs aux intersections régulières
                    if i % spacing == 0 and j % spacing == 0:
                        place_wall = True
                elif pattern == 'horizontal':
                    # Lignes horizontales avec passages
                    if i % spacing == 0 and j % (spacing * 2) != 0:
                        place_wall = True
                elif pattern == 'vertical':
                    # Colonnes verticales avec passages
                    if j % spacing == 0 and i % (spacing * 2) != 0:
                        place_wall = True
                
                if place_wall:
                    self.grid[i][j] = '#'
        
        # Replacer départ et arrivée
        self.grid[self.start[0]][self.start[1]] = 'A'
        self.grid[self.goal[0]][self.goal[1]] = 'B'
        
        # Mettre à jour les récompenses
        self.init_rewards()
    
    def init_rewards(self, move_penalty=-1, goal_reward=100, bonus_cells=None):
        """
        Initialise la matrice de récompense.
        
        Paramètres:
        - move_penalty: pénalité pour chaque déplacement (valeur négative)
        - goal_reward: récompense pour atteindre l'arrivée
        - bonus_cells: dict {(i,j): bonus} pour des cellules avec bonus
        
        La pénalité encourage les chemins courts.
        """
        if bonus_cells is None:
            bonus_cells = {}
        
        for i in range(self.height):
            for j in range(len(self.grid[i])):
                cell = self.grid[i][j]
                
                if cell == '#':
                    # Mur = coût infini (inaccessible)
                    self.rewards[i][j] = float('-inf')
                elif (i, j) == self.goal:
                    # Arrivée = grande récompense
                    self.rewards[i][j] = goal_reward
                elif (i, j) in bonus_cells:
                    # Cellule avec bonus
                    self.rewards[i][j] = bonus_cells[(i, j)]
                else:
                    # Case normale = pénalité de déplacement
                    self.rewards[i][j] = move_penalty
    
    def add_bonus(self, i, j, bonus_value):
        """
        Ajoute un bonus sur une cellule spécifique.
        
        Paramètres:
        - i, j: coordonnées de la cellule
        - bonus_value: valeur du bonus (positif)
        """
        if self.in_bounds(i, j) and self.grid[i][j] not in ('#', 'A', 'B'):
            self.rewards[i][j] = bonus_value
    
    def save_to_file(self, filename):
        """
        Sauvegarde le labyrinthe dans un fichier texte.
        """
        with open(filename, 'w') as f:
            for row in self.grid:
                f.write(''.join(row) + '\n')
    
    # ==================== FIN PARTIE B ====================
    
    # ==================== PARTIE C : Algorithme A* ====================
    
    def _heuristic(self, i, j, gi, gj, allow_diagonal=False):
        """
        Calcule l'heuristique entre (i,j) et (gi,gj).
        
        - Sans diagonale: Distance de Manhattan
        - Avec diagonale: Distance de Chebyshev (ou diagonale)
        
        Cette heuristique est admissible car elle ne surestime jamais le coût réel.
        """
        dx = abs(i - gi)
        dy = abs(j - gj)
        
        if allow_diagonal:
            # Heuristique diagonale: min(dx,dy) * sqrt(2) + |dx-dy| * 1
            return math.sqrt(2) * min(dx, dy) + abs(dx - dy)
        else:
            # Distance de Manhattan
            return dx + dy
    
    def solve(self, si, sj, gi, gj, allow_diagonal=False):
        """
        Trouve le chemin optimal de (si, sj) à (gi, gj) avec l'algorithme A*.
        
        Paramètres:
        - si, sj: position de départ (ligne, colonne)
        - gi, gj: position d'arrivée (ligne, colonne)
        - allow_diagonal: si True, autorise les déplacements diagonaux (coût sqrt(2))
        
        Retourne:
        - La liste ordonnée des cellules du chemin optimal [(i1,j1), (i2,j2), ...]
        - None si aucun chemin n'existe
        
        L'algorithme utilise:
        - Une file de priorité pour gérer les cellules à explorer
        - f(n) = g(n) + h(n) avec g = coût réel, h = heuristique (Manhattan)
        - Mémorisation des meilleurs coûts trouvés
        - Reconstruction du chemin via les relations de parenté
        """
        # g_score[cellule] = coût réel du chemin depuis le départ
        g_score = {}
        g_score[(si, sj)] = 0
        
        # f_score[cellule] = g_score + heuristique (estimation du coût total)
        f_score = {}
        f_score[(si, sj)] = self._heuristic(si, sj, gi, gj, allow_diagonal)
        
        # Pour reconstruire le chemin: parent[cellule] = cellule précédente
        parent = {}
        parent[(si, sj)] = None
        
        # File de priorité: (f_score, compteur, (i, j))
        # Le compteur sert à départager les égalités de f_score
        counter = 0
        heap = [(f_score[(si, sj)], counter, (si, sj))]
        
        # Ensemble des cellules déjà évaluées (closed set)
        visited = set()
        
        while heap:
            # Extraire la cellule avec le plus petit f_score
            current_f, _, (i, j) = heapq.heappop(heap)
            
            # Si déjà visitée, on passe
            if (i, j) in visited:
                continue
            
            # Marquer comme visitée
            visited.add((i, j))
            
            # Si on a atteint l'arrivée, reconstruire et retourner le chemin
            if (i, j) == (gi, gj):
                return self._reconstruct_path(parent, (gi, gj))
            
            # Explorer les voisins accessibles
            for (ni, nj), move_cost in self.get_neighbors(i, j, allow_diagonal):
                if (ni, nj) in visited:
                    continue
                
                # Coût pour atteindre le voisin = coût actuel + coût du déplacement
                tentative_g = g_score[(i, j)] + move_cost
                
                # Si on trouve un meilleur chemin vers ce voisin
                if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                    # Mettre à jour le meilleur coût
                    g_score[(ni, nj)] = tentative_g
                    f_score[(ni, nj)] = tentative_g + self._heuristic(ni, nj, gi, gj, allow_diagonal)
                    parent[(ni, nj)] = (i, j)
                    
                    # Ajouter à la file de priorité
                    counter += 1
                    heapq.heappush(heap, (f_score[(ni, nj)], counter, (ni, nj)))
        
        # Aucun chemin trouvé
        return None
    
    def _reconstruct_path(self, parent, goal):
        """
        Reconstruit le chemin optimal en remontant les relations de parenté.
        
        Paramètres:
        - parent: dictionnaire {cellule: cellule_précédente}
        - goal: cellule d'arrivée
        
        Retourne: liste ordonnée des cellules du départ à l'arrivée
        """
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()  # Inverser pour avoir départ -> arrivée
        return path
    
    # ==================== QUESTION 3 : Dijkstra et comparaison ====================
    
    def solve_dijkstra(self, si, sj, gi, gj, allow_diagonal=False):
        """
        Trouve le chemin optimal avec l'algorithme de Dijkstra.
        
        Dijkstra explore les cellules par ordre de coût croissant depuis le départ,
        SANS utiliser d'heuristique (contrairement à A*).
        
        Paramètres:
        - si, sj: position de départ
        - gi, gj: position d'arrivée
        - allow_diagonal: autoriser les déplacements diagonaux
        
        Retourne:
        - (path, nodes_explored): le chemin et le nombre de nœuds explorés
        - (None, nodes_explored) si aucun chemin n'existe
        """
        # Distance minimale pour atteindre chaque cellule
        dist = {}
        dist[(si, sj)] = 0
        
        # Pour reconstruire le chemin
        parent = {}
        parent[(si, sj)] = None
        
        # File de priorité: (distance, compteur, (i, j))
        counter = 0
        heap = [(0, counter, (si, sj))]
        
        # Cellules visitées
        visited = set()
        nodes_explored = 0
        
        while heap:
            current_dist, _, (i, j) = heapq.heappop(heap)
            
            if (i, j) in visited:
                continue
            
            visited.add((i, j))
            nodes_explored += 1
            
            # Arrivée atteinte
            if (i, j) == (gi, gj):
                return self._reconstruct_path(parent, (gi, gj)), nodes_explored
            
            # Explorer les voisins
            for (ni, nj), move_cost in self.get_neighbors(i, j, allow_diagonal):
                if (ni, nj) in visited:
                    continue
                
                new_dist = current_dist + move_cost
                
                if (ni, nj) not in dist or new_dist < dist[(ni, nj)]:
                    dist[(ni, nj)] = new_dist
                    parent[(ni, nj)] = (i, j)
                    counter += 1
                    heapq.heappush(heap, (new_dist, counter, (ni, nj)))
        
        return None, nodes_explored
    
    def solve_astar(self, si, sj, gi, gj, allow_diagonal=False):
        """
        Trouve le chemin optimal avec l'algorithme A*.
        
        A* utilise une heuristique pour guider la recherche vers l'arrivée,
        ce qui réduit généralement le nombre de nœuds explorés.
        
        Paramètres:
        - si, sj: position de départ
        - gi, gj: position d'arrivée
        - allow_diagonal: autoriser les déplacements diagonaux
        
        Retourne:
        - (path, nodes_explored): le chemin et le nombre de nœuds explorés
        - (None, nodes_explored) si aucun chemin n'existe
        """
        g_score = {}
        g_score[(si, sj)] = 0
        
        f_score = {}
        f_score[(si, sj)] = self._heuristic(si, sj, gi, gj, allow_diagonal)
        
        parent = {}
        parent[(si, sj)] = None
        
        counter = 0
        heap = [(f_score[(si, sj)], counter, (si, sj))]
        
        visited = set()
        nodes_explored = 0
        
        while heap:
            current_f, _, (i, j) = heapq.heappop(heap)
            
            if (i, j) in visited:
                continue
            
            visited.add((i, j))
            nodes_explored += 1
            
            if (i, j) == (gi, gj):
                return self._reconstruct_path(parent, (gi, gj)), nodes_explored
            
            for (ni, nj), move_cost in self.get_neighbors(i, j, allow_diagonal):
                if (ni, nj) in visited:
                    continue
                
                tentative_g = g_score[(i, j)] + move_cost
                
                if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                    g_score[(ni, nj)] = tentative_g
                    f_score[(ni, nj)] = tentative_g + self._heuristic(ni, nj, gi, gj, allow_diagonal)
                    parent[(ni, nj)] = (i, j)
                    counter += 1
                    heapq.heappush(heap, (f_score[(ni, nj)], counter, (ni, nj)))
        
        return None, nodes_explored
    
    def compare_algorithms(self, allow_diagonal=False, verbose=True):
        """
        Compare Dijkstra et A* sur ce labyrinthe.
        
        Retourne un dictionnaire avec les métriques de comparaison.
        """
        import time
        
        si, sj = self.start
        gi, gj = self.goal
        
        # --- Dijkstra ---
        start_time = time.perf_counter()
        path_dijkstra, nodes_dijkstra = self.solve_dijkstra(si, sj, gi, gj, allow_diagonal)
        time_dijkstra = (time.perf_counter() - start_time) * 1000  # ms
        
        # --- A* ---
        start_time = time.perf_counter()
        path_astar, nodes_astar = self.solve_astar(si, sj, gi, gj, allow_diagonal)
        time_astar = (time.perf_counter() - start_time) * 1000  # ms
        
        results = {
            'dijkstra': {
                'path_length': len(path_dijkstra) if path_dijkstra else 0,
                'nodes_explored': nodes_dijkstra,
                'time_ms': time_dijkstra,
                'path': path_dijkstra
            },
            'astar': {
                'path_length': len(path_astar) if path_astar else 0,
                'nodes_explored': nodes_astar,
                'time_ms': time_astar,
                'path': path_astar
            }
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("COMPARAISON DIJKSTRA vs A*")
            print("=" * 60)
            print(f"Labyrinthe: {self.height}x{self.width}")
            print(f"Diagonales: {'Oui' if allow_diagonal else 'Non'}")
            print("-" * 60)
            print(f"{'Métrique':<25} {'Dijkstra':>15} {'A*':>15}")
            print("-" * 60)
            print(f"{'Longueur du chemin':<25} {results['dijkstra']['path_length']:>15} {results['astar']['path_length']:>15}")
            print(f"{'Nœuds explorés':<25} {results['dijkstra']['nodes_explored']:>15} {results['astar']['nodes_explored']:>15}")
            print(f"{'Temps (ms)':<25} {results['dijkstra']['time_ms']:>15.3f} {results['astar']['time_ms']:>15.3f}")
            print("-" * 60)
            
            # Calcul des gains
            if nodes_dijkstra > 0:
                gain_nodes = ((nodes_dijkstra - nodes_astar) / nodes_dijkstra) * 100
                print(f"{'Gain A* (nœuds)':<25} {gain_nodes:>15.1f}%")
            if time_dijkstra > 0:
                gain_time = ((time_dijkstra - time_astar) / time_dijkstra) * 100
                print(f"{'Gain A* (temps)':<25} {gain_time:>15.1f}%")
            print("=" * 60)
            
            # Analyse
            print("\nANALYSE:")
            print("-" * 40)
            if path_dijkstra and path_astar:
                if len(path_dijkstra) == len(path_astar):
                    print("✓ Les deux algorithmes trouvent un chemin de même longueur")
                    print("  (les deux sont optimaux)")
                else:
                    print("⚠ Longueurs différentes (vérifier l'implémentation)")
                
                if nodes_astar < nodes_dijkstra:
                    print(f"✓ A* explore {nodes_dijkstra - nodes_astar} nœuds de moins ({gain_nodes:.1f}% de gain)")
                    print("  grâce à l'heuristique qui guide vers l'arrivée")
                elif nodes_astar == nodes_dijkstra:
                    print("→ A* et Dijkstra explorent le même nombre de nœuds")
                    print("  (peut arriver si l'heuristique n'aide pas)")
                else:
                    print("⚠ A* explore plus de nœuds (cas rare)")
            elif not path_dijkstra and not path_astar:
                print("✗ Aucun chemin trouvé par les deux algorithmes")
            else:
                print("⚠ Un algorithme trouve un chemin, l'autre non (erreur)")
        
        return results
    
    def compare_graphical(self, allow_diagonal=False):
        """
        Affiche une comparaison graphique entre Dijkstra et A*.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Erreur: matplotlib n'est pas installé.")
            return
        
        si, sj = self.start
        gi, gj = self.goal
        
        path_dijkstra, nodes_dijkstra = self.solve_dijkstra(si, sj, gi, gj, allow_diagonal)
        path_astar, nodes_astar = self.solve_astar(si, sj, gi, gj, allow_diagonal)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        for ax, path, title, nodes in [
            (ax1, path_dijkstra, f"DIJKSTRA\n({nodes_dijkstra} nœuds explorés)", nodes_dijkstra),
            (ax2, path_astar, f"A*\n({nodes_astar} nœuds explorés)", nodes_astar)
        ]:
            matrix = np.zeros((self.height, self.width))
            
            for i in range(self.height):
                for j in range(len(self.grid[i])):
                    cell = self.grid[i][j]
                    if cell == '#':
                        matrix[i][j] = 1
                    elif cell == 'A':
                        matrix[i][j] = 3
                    elif cell == 'B':
                        matrix[i][j] = 4
            
            if path:
                for (i, j) in path:
                    if matrix[i][j] == 0:
                        matrix[i][j] = 2
            
            colors = ['white', 'black', 'limegreen', 'dodgerblue', 'red']
            cmap = ListedColormap(colors)
            
            ax.imshow(matrix, cmap=cmap, vmin=0, vmax=4)
            ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            
            if self.start:
                ax.text(self.start[1], self.start[0], 'A', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
            if self.goal:
                ax.text(self.goal[1], self.goal[0], 'B', ha='center', va='center',
                       fontsize=12, fontweight='bold', color='white')
            
            if path and len(path) > 1:
                path_y = [p[0] for p in path]
                path_x = [p[1] for p in path]
                ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.7)
            
            length_info = f"Chemin: {len(path)} cellules" if path else "Pas de chemin"
            ax.set_title(f"{title}\n{length_info}", fontsize=12, fontweight='bold')
        
        fig.suptitle("Comparaison DIJKSTRA vs A*", fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}/compare_dijkstra_astar.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Image sauvegardée: {filename}")
    
    # ==================== FIN QUESTION 3 ====================
    
    # ==================== FIN PARTIE C ====================
    
    def display(self, path=None):
        """
        Affiche le labyrinthe avec des couleurs.
        - Chemin: vert
        - Départ (A): bleu
        - Arrivée (B): rouge
        - Murs (#): gris
        - Cases libres (.): blanc
        """
        # Codes couleur ANSI
        RESET = '\033[0m'
        GREEN = '\033[92m'   # Chemin
        BLUE = '\033[94m'    # Départ
        RED = '\033[91m'     # Arrivée
        GRAY = '\033[90m'    # Murs
        YELLOW = '\033[93m'  # Symbole du chemin
        
        # Convertir le chemin en set pour recherche rapide
        path_set = set(path) if path else set()
        
        for i in range(self.height):
            line = ""
            for j in range(len(self.grid[i])):  # Utiliser la longueur réelle de chaque ligne
                cell = self.grid[i][j]
                
                if (i, j) in path_set:
                    if cell == 'A':
                        line += BLUE + 'A' + RESET
                    elif cell == 'B':
                        line += RED + 'B' + RESET
                    else:
                        line += GREEN + '●' + RESET  # Symbole du chemin
                elif cell == '#':
                    line += GRAY + '█' + RESET
                elif cell == 'A':
                    line += BLUE + 'A' + RESET
                elif cell == 'B':
                    line += RED + 'B' + RESET
                else:
                    line += ' '  # Case libre = espace
            print(line)
    
    # ==================== QUESTION 2 : Visualisation graphique ====================
    
    def display_graphical(self, path=None, title="Labyrinthe", show_grid=True, save_file=None):
        """
        Affiche le labyrinthe avec une visualisation graphique (matplotlib).
        
        Paramètres:
        - path: liste des cellules du chemin (optionnel)
        - title: titre de la figure
        - show_grid: afficher la grille
        - save_file: chemin pour sauvegarder l'image (optionnel)
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Erreur: matplotlib n'est pas installé.")
            print("Installer avec: pip install matplotlib numpy")
            return
        
        # Créer une matrice numérique pour la visualisation
        # 0 = case libre, 1 = mur, 2 = chemin, 3 = départ, 4 = arrivée
        matrix = np.zeros((self.height, self.width))
        
        for i in range(self.height):
            for j in range(len(self.grid[i])):
                cell = self.grid[i][j]
                if cell == '#':
                    matrix[i][j] = 1  # Mur
                elif cell == 'A':
                    matrix[i][j] = 3  # Départ
                elif cell == 'B':
                    matrix[i][j] = 4  # Arrivée
        
        # Marquer le chemin
        if path:
            for (i, j) in path:
                if matrix[i][j] == 0:  # Ne pas écraser départ/arrivée
                    matrix[i][j] = 2  # Chemin
        
        # Définir les couleurs
        # 0=blanc (libre), 1=noir (mur), 2=vert (chemin), 3=bleu (départ), 4=rouge (arrivée)
        colors = ['white', 'black', 'limegreen', 'dodgerblue', 'red']
        cmap = ListedColormap(colors)
        
        # Créer la figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=4)
        
        # Ajouter la grille
        if show_grid:
            ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Marquer départ et arrivée avec des labels
        if self.start:
            ax.text(self.start[1], self.start[0], 'A', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        if self.goal:
            ax.text(self.goal[1], self.goal[0], 'B', ha='center', va='center', 
                   fontsize=14, fontweight='bold', color='white')
        
        # Dessiner le chemin avec des lignes si fourni
        if path and len(path) > 1:
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.7)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Colonne (j)')
        ax.set_ylabel('Ligne (i)')
        
        # Afficher la longueur du chemin si disponible
        if path:
            ax.text(0.02, 0.98, f'Longueur: {len(path)} cellules', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Image sauvegardée: {save_file}")
        else:
            filename = f"{OUTPUT_DIR}/{title.replace(' ', '_').replace('*', 'star')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Image sauvegardée: {filename}")
        
        plt.close()
    
    def compare_paths(self, title="Comparaison: 4 directions vs 8 directions (diagonales)"):
        """
        Compare visuellement les chemins avec et sans déplacements diagonaux.
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Erreur: matplotlib n'est pas installé.")
            return
        
        # Résoudre avec 4 directions
        path_4dir = self.solve(self.start[0], self.start[1], 
                               self.goal[0], self.goal[1], allow_diagonal=False)
        
        # Résoudre avec 8 directions (diagonales)
        path_8dir = self.solve(self.start[0], self.start[1], 
                               self.goal[0], self.goal[1], allow_diagonal=True)
        
        # Créer une figure avec 2 sous-graphes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Préparer les matrices
        for ax, path, subtitle, directions in [
            (ax1, path_4dir, "4 directions (Manhattan)", "4 dir"),
            (ax2, path_8dir, "8 directions (Diagonales)", "8 dir")
        ]:
            matrix = np.zeros((self.height, self.width))
            
            for i in range(self.height):
                for j in range(len(self.grid[i])):
                    cell = self.grid[i][j]
                    if cell == '#':
                        matrix[i][j] = 1
                    elif cell == 'A':
                        matrix[i][j] = 3
                    elif cell == 'B':
                        matrix[i][j] = 4
            
            if path:
                for (i, j) in path:
                    if matrix[i][j] == 0:
                        matrix[i][j] = 2
            
            colors = ['white', 'black', 'limegreen', 'dodgerblue', 'red']
            cmap = ListedColormap(colors)
            
            ax.imshow(matrix, cmap=cmap, vmin=0, vmax=4)
            ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
            
            if self.start:
                ax.text(self.start[1], self.start[0], 'A', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
            if self.goal:
                ax.text(self.goal[1], self.goal[0], 'B', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color='white')
            
            if path and len(path) > 1:
                path_y = [p[0] for p in path]
                path_x = [p[1] for p in path]
                ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.7)
            
            length_info = f"Cellules: {len(path)}" if path else "Pas de chemin"
            ax.set_title(f"{subtitle}\n{length_info}", fontsize=12, fontweight='bold')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        filename = f"{OUTPUT_DIR}/compare_4dir_8dir.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Image sauvegardée: {filename}")
    
    # ==================== FIN QUESTION 2 ====================
    
    # ==================== QUESTION 4 : Test avec poids négatifs ====================
    
    def test_negative_weights(self):
        """
        Question 4: Teste les algorithmes Dijkstra et A* avec des poids négatifs.
        Retourne les résultats pour analyse dans le compte rendu.
        """
        # Créer un labyrinthe avec des poids négatifs (bonus)
        maze = Maze.create_empty(10, 10, start=(1, 1), goal=(8, 8))
        negative_cells = [(3, 3), (4, 4), (5, 5), (6, 6)]
        for (i, j) in negative_cells:
            maze.rewards[i][j] = -2
        
        results = {
            'maze_size': (maze.height, maze.width),
            'negative_cells': negative_cells,
            'dijkstra': None,
            'astar': None
        }
        
        # Test Dijkstra
        try:
            path_dij, nodes_dij = maze.solve_dijkstra(
                maze.start[0], maze.start[1], maze.goal[0], maze.goal[1])
            results['dijkstra'] = {'path_len': len(path_dij) if path_dij else 0, 'nodes': nodes_dij}
        except Exception as e:
            results['dijkstra'] = {'error': str(e)}
        
        # Test A*
        try:
            path_astar, nodes_astar = maze.solve_astar(
                maze.start[0], maze.start[1], maze.goal[0], maze.goal[1])
            results['astar'] = {'path_len': len(path_astar) if path_astar else 0, 'nodes': nodes_astar}
        except Exception as e:
            results['astar'] = {'error': str(e)}
        
        return results


# Code de test
if __name__ == "__main__":
    # Charger le labyrinthe
    maze = Maze("labyrinthe.txt")
    maze.display()
    
    # Créer labyrinthe avec obstacles aléatoires
    maze2 = Maze.create_empty(25, 25, start=(1, 1), goal=(23, 23))
    maze2.generate_obstacles_random(density=0.15, seed=42)
    
    # Créer labyrinthe avec obstacles déterministes
    maze3 = Maze.create_empty(25, 25, start=(1, 1), goal=(23, 23))
    maze3.generate_obstacles_deterministic(pattern='grid', spacing=5)
    
    # Résolution A*
    path = maze.solve(maze.start[0], maze.start[1], maze.goal[0], maze.goal[1])
    if path:
        maze.display(path)
    
    # Résolution Dijkstra
    path_dij, _ = maze.solve_dijkstra(maze.start[0], maze.start[1], maze.goal[0], maze.goal[1])
    
    # Visualisation graphique
    if MATPLOTLIB_AVAILABLE:
        maze.display_graphical(path, title="A*")
        maze.display_graphical(path_dij, title="Dijkstra")
        
        path2 = maze2.solve(maze2.start[0], maze2.start[1], maze2.goal[0], maze2.goal[1])
        maze2.display_graphical(path2, title="Labyrinthe aléatoire - A*")
        
        path3 = maze3.solve(maze3.start[0], maze3.start[1], maze3.goal[0], maze3.goal[1])
        maze3.display_graphical(path3, title="Labyrinthe déterministe - A*")
    
    # Tests de validation Question 1
    maze_empty = Maze.create_empty(15, 15, start=(1, 1), goal=(13, 13))
    path_empty = maze_empty.solve(maze_empty.start[0], maze_empty.start[1], 
                                   maze_empty.goal[0], maze_empty.goal[1])
    
    maze_simple = Maze.create_empty(15, 15, start=(1, 1), goal=(13, 13))
    for i in range(2, 12):
        maze_simple.grid[i][7] = '#'
    maze_simple.grid[6][7] = '.'
    maze_simple.init_rewards()
    path_simple = maze_simple.solve(maze_simple.start[0], maze_simple.start[1],
                                     maze_simple.goal[0], maze_simple.goal[1])
    
    maze_blocked = Maze.create_empty(15, 15, start=(1, 1), goal=(13, 13))
    for i in range(1, 14):
        maze_blocked.grid[i][7] = '#'
    maze_blocked.init_rewards()
    path_blocked = maze_blocked.solve(maze_blocked.start[0], maze_blocked.start[1],
                                       maze_blocked.goal[0], maze_blocked.goal[1])
    
    # Question 2: Diagonales

    # Génération des images pour le compte rendu universitaire (Tests 1, 2, 3)
    if MATPLOTLIB_AVAILABLE:
        # Test 1 : Labyrinthe vide
        maze_empty.display_graphical(path_empty, title="Labyrinthe vide (Test 1)", save_file="output_images/labyrinthe_vide_test1.png")
        # Test 2 : Obstacles simples
        maze_simple.display_graphical(path_simple, title="Labyrinthe obstacles simples (Test 2)", save_file="output_images/labyrinthe_obstacles_test2.png")
        # Test 3 : Labyrinthe bloqué
        maze_blocked.display_graphical(path_blocked, title="Labyrinthe bloqué (Test 3)", save_file="output_images/labyrinthe_bloque_test3.png")
    maze_diag = Maze.create_empty(20, 20, start=(1, 1), goal=(18, 18))
    maze_diag.generate_obstacles_random(density=0.1, seed=123)
    
    path_4dir = maze_diag.solve(maze_diag.start[0], maze_diag.start[1],
                                 maze_diag.goal[0], maze_diag.goal[1], allow_diagonal=False)
    path_8dir = maze_diag.solve(maze_diag.start[0], maze_diag.start[1],
                                 maze_diag.goal[0], maze_diag.goal[1], allow_diagonal=True)
    
    if MATPLOTLIB_AVAILABLE:
        maze_diag.display_graphical(path_4dir, title="A* (4 directions)")
        maze_diag.display_graphical(path_8dir, title="A* (8 directions)")
        maze_diag.compare_paths()
    
    # Question 3: Comparaison Dijkstra vs A*
    maze_small = Maze.create_empty(15, 15, start=(1, 1), goal=(13, 13))
    maze_small.generate_obstacles_random(density=0.15, seed=42)
    results_small = maze_small.compare_algorithms(allow_diagonal=False, verbose=False)
    
    maze_medium = Maze.create_empty(30, 30, start=(1, 1), goal=(28, 28))
    maze_medium.generate_obstacles_random(density=0.2, seed=123)
    results_medium = maze_medium.compare_algorithms(allow_diagonal=False, verbose=False)
    
    maze_large = Maze.create_empty(50, 50, start=(1, 1), goal=(48, 48))
    maze_large.generate_obstacles_random(density=0.25, seed=456)
    results_large = maze_large.compare_algorithms(allow_diagonal=False, verbose=False)
    
    results_medium_diag = maze_medium.compare_algorithms(allow_diagonal=True, verbose=False)
    
    if MATPLOTLIB_AVAILABLE:
        maze_small.compare_graphical(allow_diagonal=False)
        maze_medium.compare_graphical(allow_diagonal=False)
        maze_medium.compare_graphical(allow_diagonal=True)
    
    # Question 4: Test avec poids négatifs 
    results_q4 = maze.test_negative_weights()

