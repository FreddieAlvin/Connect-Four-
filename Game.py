import pygame
import sys
import random
import math
import copy

# Inicialização do Pygame
pygame.init()

# Configurações da tela e do jogo
SCREEN_WIDTH, SCREEN_HEIGHT = 700, 600  # 7 colunas x 6 linhas, 100px cada
BG_COLOR = (0, 0, 0)
COLORS = {
    "BLUE": (0, 0, 255),
    "RED": (255, 0, 0),
    "YELLOW": (255, 255, 0)
}
COLS = 7
ROWS = 6
SQUARE_SIZE = 100

# Fontes para textos
font_title = pygame.font.Font(None, 60)
font_button = pygame.font.Font(None, 40)

# --- Classe do Jogo Connect Four ---
class ConnectFourGame:
    def __init__(self):
        # Cria um tabuleiro vazio com 6 linhas e 7 colunas
        self.board = [['' for _ in range(COLS)] for _ in range(ROWS)]
        self.current_turn = 'red'  # Jogador inicial
        self.move_log = []

    def get_legal_moves(self):
        # No Connect Four, um movimento válido implica escolher uma coluna não cheia.
        legal_moves = []
        for col in range(COLS):
            if self.board[0][col] == '':
                legal_moves.append(col)
        return legal_moves

    def make_move(self, col):
        if col not in self.get_legal_moves():
            return False
        # A peça "cai" para a linha mais baixa disponível na coluna.
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == '':
                self.board[row][col] = self.current_turn
                self.move_log.append((row, col, self.current_turn))
                # Alterna o turno do jogador.
                self.current_turn = 'red' if self.current_turn == 'yellow' else 'yellow'
                break
        return True

    def undo_move(self):
        if self.move_log:
            row, col, player = self.move_log.pop()
            self.board[row][col] = ''
            self.current_turn = player

    def is_game_over(self):
        return self.check_winner or (self.get_legal_moves()) == 0

    def check_winner(self):
        # Verifica vitórias horizontais
        for row in range(ROWS):
            for col in range(COLS - 3):
                token = self.board[row][col]
                if token != '' and all(self.board[row][col + i] == token for i in range(4)):
                    return token
        # Verifica vitórias verticais
        for col in range(COLS):
            for row in range(ROWS - 3):
                token = self.board[row][col]
                if token != '' and all(self.board[row + i][col] == token for i in range(4)):
                    return token
        # Diagonais (descendo para a direita)
        for row in range(ROWS - 3):
            for col in range(COLS - 3):
                token = self.board[row][col]
                if token != '' and all(self.board[row + i][col + i] == token for i in range(4)):
                    return token
        # Diagonais (subindo para a direita)
        for row in range(3, ROWS):
            for col in range(COLS - 3):
                token = self.board[row][col]
                if token != '' and all(self.board[row - i][col + i] == token for i in range(4)):
                    return token
        return None

    def draw(self, screen):
        # Desenha o tabuleiro (fundo azul) e as peças (círculos)
        for row in range(ROWS):
            for col in range(COLS):
                # Desenha o retângulo azul para a célula
                pygame.draw.rect(screen, COLORS["BLUE"], (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                # Desenha o "buraco" (círculo) da célula
                pygame.draw.circle(screen, BG_COLOR, (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                       row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                                       SQUARE_SIZE // 2 - 5)
                # Se houver uma peça, desenha-a (vermelha ou amarela)
                if self.board[row][col] == 'red':
                    pygame.draw.circle(screen, COLORS["RED"], (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                     row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                                     SQUARE_SIZE // 2 - 5)
                elif self.board[row][col] == 'yellow':
                    pygame.draw.circle(screen, COLORS["YELLOW"], (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                                        row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                                        SQUARE_SIZE // 2 - 5)

# --- Implementação do MCTS com UCT ---
EXPLORATION_CONSTANT = 1.41

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state                  # Estado do jogo (instância de ConnectFourGame)
        self.parent = parent                # Nó pai na árvore
        self.move = move                    # Movimento que levou a este nó
        self.children = []                  # Lista de nós filhos
        self.wins = 0                       # Número de vitórias obtidas
        self.visits = 0                     # Número de vezes que o nó foi visitado
        self.untried_moves = state.get_legal_moves()  # Movimentos ainda não explorados a partir deste estado

    def uct_value(self, total_simulations):
        # Se o nó não foi visitado, retorna um valor muito alto para incentivar a exploração.
        if self.visits == 0:
            return float('inf')
        exploitation = self.wins / self.visits
        exploration = EXPLORATION_CONSTANT * math.sqrt(math.log(total_simulations) / self.visits)
        return exploitation + exploration

def mcts(root_state, itermax):
    root_node = MCTSNode(root_state)
    for _ in range(itermax):
        node = root_node
        state = copy.deepcopy(root_state)

        # 1. Seleção: Navega pela árvore utilizando UCT até encontrar um nó não totalmente expandido.
        while node.untried_moves == [] and node.children:
            node = max(node.children, key=lambda child: child.uct_value(node.visits))
            state.make_move(node.move)

        # 2. Expansão: Se houver movimentos não explorados, expande um deles.
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            state.make_move(move)
            child_node = MCTSNode(copy.deepcopy(state), parent=node, move=move)
            node.children.append(child_node)
            node.untried_moves.remove(move)
            node = child_node

        # 3. Simulação (Rollout): A partir do nó expandido, simula uma partida aleatória até o fim.
        while not state.is_game_over():
            legal_moves = state.get_legal_moves()
            if not legal_moves:
                break
            state.make_move(random.choice(legal_moves))

        # 4. Backpropagation: Atualiza as estatísticas de todos os nós visitados com o resultado da simulação.
        winner = state.check_winner()
        while node is not None:
            node.visits += 1
            # Se o vencedor for o mesmo que o jogador da raiz, conta como vitória para esse nó.
            # (Note que o valor de current_turn da raiz representa o jogador que estava para jogar,
            # então você pode ajustar esse critério conforme a sua perspectiva.)
            if winner is not None:
                original_player = root_state.current_turn
                if winner == original_player:
                    node.wins += 1
            node = node.parent

    # Após as iterações, seleciona o movimento do nó filho com o maior número de visitas.
    best_move = max(root_node.children, key=lambda child: child.visits).move
    return best_move

def get_computer_move(game, itermax=1000):
    # Utiliza o MCTS para escolher o movimento.
    return mcts(game, itermax)

# --- Funções de interface e modos de jogo ---
def game_mode_selection_screen(screen):
    buttons = ['User vs User', 'User vs Computer', 'Computer vs Computer']
    num_buttons = len(buttons)
    button_width = 300
    button_height = 50
    vertical_spacing = 30
    total_height = num_buttons * button_height + (num_buttons - 1) * vertical_spacing
    start_y = (SCREEN_HEIGHT - total_height) // 2
    button_rects = [pygame.Rect((SCREEN_WIDTH - button_width)//2,
                                start_y + i * (button_height + vertical_spacing),
                                button_width, button_height) for i in range(num_buttons)]
    running = True
    while running:
        screen.fill(BG_COLOR)
        title = font_title.render("Selecione o modo de jogo", True, (255, 255, 255))
        title_rect = title.get_rect(center=(SCREEN_WIDTH//2, start_y - 60))
        screen.blit(title, title_rect)
        for i, rect in enumerate(button_rects):
            pygame.draw.rect(screen, (235, 26, 235), rect)
            text = font_button.render(buttons[i], True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            screen.blit(text, text_rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                for i, rect in enumerate(button_rects):
                    if rect.collidepoint(pos):
                        if buttons[i] == 'User vs User':
                            return "user_vs_user"
                        elif buttons[i] == 'User vs Computer':
                            return "user_vs_computer"
                        else:
                            return "computer_vs_computer"

def update_screen(screen, game):
    screen.fill(BG_COLOR)
    game.draw(screen)
    pygame.display.flip()

def game_screen(screen, mode):
    game = ConnectFourGame()
    running = True
    player_turn = True  # Indica a vez do jogador (no modo User vs Computer)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            # Se o modo permitir input do usuário:
            if mode != "computer_vs_computer" and event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x = event.pos[0]
                col = mouse_x // SQUARE_SIZE
                if col in game.get_legal_moves():
                    game.make_move(col)
                    player_turn = False
        
        # Jogada do computador:
        if mode == "user_vs_computer" and not player_turn and not game.is_game_over():
            pygame.time.wait(500)
            comp_move = get_computer_move(game, itermax=500)
            game.make_move(comp_move)
            player_turn = True
        elif mode == "computer_vs_computer" and not game.is_game_over():
            pygame.time.wait(500)
            move = get_computer_move(game, itermax=500)
            game.make_move(move)
        
        update_screen(screen, game)
        
        if game.is_game_over():
            winner = game.check_winner()
            if winner:
                result_text = f"Vencedor: {winner.upper()}"
            else:
                result_text = "Empate!"
            text_surface = font_title.render(result_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            screen.blit(text_surface, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)
            running = False

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Connect Four com MCTS")
    mode = game_mode_selection_screen(screen)
    game_screen(screen, mode)

if __name__ == "__main__":
    main()