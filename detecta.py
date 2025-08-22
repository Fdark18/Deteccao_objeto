#!/usr/bin/env python3
"""
Sistema de Detecção de Dispositivos Móveis usando YOLO
===============================================

Descrição:
    Sistema profissional para detecção em tempo real de dispositivos móveis
    (celulares, tablets, laptops) utilizando algoritmos de visão computacional
    baseados na arquitetura YOLO (You Only Look Once).

Autor: Sistema de Visão Computacional
Data: 2025
Versão: 1.0.0

Dependências:
    - ultralytics >= 8.0.0
    - opencv-python >= 4.8.0
    - numpy >= 1.24.0
    - torch >= 2.0.0

Uso:
    python detector_celular_strict.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, cast

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    print("ERRO: Biblioteca ultralytics não encontrada.")
    print("Execute: pip install ultralytics")
    sys.exit(1)


# Type aliases para melhor legibilidade
BoundingBox = Tuple[int, int, int, int]
ColorBGR = Tuple[int, int, int]
DetectionDict = Dict[str, Union[BoundingBox, float, int, str]]


class DetectorDispositivos:
    """
    Classe principal para detecção de dispositivos móveis em tempo real.
    
    Esta classe implementa um sistema de detecção baseado em YOLO capaz de
    identificar diferentes tipos de dispositivos eletrônicos em stream de vídeo.
    
    Attributes:
        model: Modelo YOLO carregado para inferência
        conf_threshold: Limite de confiança para detecções válidas
        classes_interesse: Mapeamento de IDs de classe para nomes
        cores_deteccao: Cores BGR para visualização de cada classe
        total_deteccoes: Contador total de detecções na sessão
        frames_processados: Contador de frames processados
        logger: Logger para registro de eventos
    """
    
    def __init__(
        self, 
        model_path: str = 'yolov8n.pt', 
        conf_threshold: float = 0.5
    ) -> None:
        """
        Inicializa o detector com parâmetros especificados.
        
        Args:
            model_path: Caminho para o arquivo do modelo YOLO
            conf_threshold: Threshold de confiança (0.0 - 1.0)
            
        Raises:
            SystemExit: Se não for possível carregar o modelo
        """
        self._configurar_logging()
        self.logger: logging.Logger = logging.getLogger(__name__)
        
        self.logger.info("Inicializando sistema de detecção de dispositivos")
        
        # Carregamento do modelo YOLO
        self.model: YOLO = self._carregar_modelo(model_path)
        self.conf_threshold: float = conf_threshold
        
        # Definição das classes de interesse baseadas no dataset COCO
        self.classes_interesse: Dict[int, str] = {
            67: "Celular",      # cell phone
            76: "Laptop",       # laptop
            72: "TV",           # tv
            73: "Mouse",        # mouse
            74: "Controle"      # remote
        }
        
        # Configuração de cores para visualização (formato BGR)
        self.cores_deteccao: Dict[int, ColorBGR] = {
            67: (0, 255, 0),    # Verde para celular
            76: (255, 0, 0),    # Azul para laptop
            72: (0, 0, 255),    # Vermelho para TV
            73: (255, 255, 0),  # Ciano para mouse
            74: (255, 0, 255)   # Magenta para controle
        }
        
        # Estatísticas de sessão
        self.total_deteccoes: int = 0
        self.frames_processados: int = 0
        
        self.logger.info("Sistema inicializado com sucesso")
    
    def _configurar_logging(self) -> None:
        """
        Configura o sistema de logging para o módulo.
        
        Cria um logger formatado para saída em console e arquivo,
        com níveis apropriados para debugging e monitoramento.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('detector_dispositivos.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _carregar_modelo(self, model_path: str) -> YOLO:
        """
        Carrega o modelo YOLO especificado.
        
        Args:
            model_path: Caminho para o modelo
            
        Returns:
            Instância do modelo YOLO carregado
            
        Raises:
            SystemExit: Se falhar ao carregar qualquer modelo
        """
        try:
            modelo = YOLO(model_path)
            self.logger.info(f"Modelo {model_path} carregado com sucesso")
            return modelo
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo {model_path}: {e}")
            self.logger.info("Tentando baixar modelo padrão YOLOv8n...")
            try:
                modelo = YOLO('yolov8n.pt')
                self.logger.info("Modelo YOLOv8n carregado com sucesso")
                return modelo
            except Exception as e:
                self.logger.critical(f"Falha crítica ao carregar modelo: {e}")
                sys.exit(1)
    
    def processar_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionDict]]:
        """
        Processa um frame individual para detecção de dispositivos.
        
        Args:
            frame: Frame de entrada no formato BGR
            
        Returns:
            Tupla contendo:
                - Frame processado com anotações
                - Lista de dicionários com detecções
        """
        deteccoes: List[DetectionDict] = []
        frame_anotado: np.ndarray = frame.copy()
        
        try:
            # Execução da inferência YOLO
            results: List[Results] = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extração dos parâmetros da caixa delimitadora
                        coordenadas = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, coordenadas)
                        bbox: BoundingBox = (x1, y1, x2, y2)
                        
                        confianca: float = float(box.conf[0].cpu().numpy())
                        classe_id: int = int(box.cls[0].cpu().numpy())
                        
                        # Filtro por classes de interesse
                        if classe_id in self.classes_interesse:
                            deteccao: DetectionDict = {
                                'bbox': bbox,
                                'confianca': confianca,
                                'classe_id': classe_id,
                                'classe_nome': self.classes_interesse[classe_id],
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            deteccoes.append(deteccao)
                            self.total_deteccoes += 1
                            
                            # Desenho da caixa delimitadora
                            frame_anotado = self._desenhar_deteccao(
                                frame_anotado, bbox, confianca, classe_id
                            )
            
            self.frames_processados += 1
            
        except Exception as e:
            self.logger.error(f"Erro no processamento do frame: {e}")
        
        return frame_anotado, deteccoes
    
    def _desenhar_deteccao(
        self, 
        frame: np.ndarray, 
        bbox: BoundingBox, 
        confianca: float, 
        classe_id: int
    ) -> np.ndarray:
        """
        Desenha uma detecção específica no frame.
        
        Args:
            frame: Frame onde desenhar
            bbox: Coordenadas da caixa delimitadora
            confianca: Nível de confiança da detecção
            classe_id: ID da classe detectada
            
        Returns:
            Frame com a detecção desenhada
        """
        x1, y1, x2, y2 = bbox
        cor: ColorBGR = self.cores_deteccao.get(classe_id, (128, 128, 128))
        
        # Desenho da caixa delimitadora
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
        
        # Preparação do texto da label
        nome_classe: str = self.classes_interesse.get(classe_id, f"Classe_{classe_id}")
        label: str = f"{nome_classe}: {confianca:.2f}"
        
        # Cálculo das dimensões do texto
        font: int = cv2.FONT_HERSHEY_SIMPLEX
        font_scale: float = 0.6
        thickness: int = 2
        
        text_size: Tuple[Tuple[int, int], int] = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        (text_width, text_height), baseline = text_size
        
        # Desenho do fundo da label
        cv2.rectangle(
            frame,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            cor,
            cv2.FILLED
        )
        
        # Desenho do texto da label
        cv2.putText(
            frame,
            label,
            (x1, y1 - baseline - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
        
        return frame
    
    def _desenhar_interface_informacoes(
        self, 
        frame: np.ndarray, 
        deteccoes: List[DetectionDict], 
        fps: float, 
        status: str
    ) -> np.ndarray:
        """
        Desenha interface com informações do sistema no frame.
        
        Args:
            frame: Frame base para desenho
            deteccoes: Lista de detecções atuais
            fps: Taxa de frames por segundo atual
            status: Status atual do sistema
            
        Returns:
            Frame com interface de informações
        """
        altura, largura = frame.shape[:2]
        
        # Criação do painel de informações
        painel_altura: int = 100
        painel: np.ndarray = np.zeros((painel_altura, largura, 3), dtype=np.uint8)
        painel.fill(45)  # Cor de fundo cinza escuro
        
        # Contagem por tipo de dispositivo
        contadores: Dict[int, int] = {}
        for classe_id in self.classes_interesse.keys():
            contadores[classe_id] = sum(
                1 for det in deteccoes 
                if cast(int, det['classe_id']) == classe_id
            )
        
        # Informações do sistema
        info_linha1: str = f"Deteccoes Ativas: {len(deteccoes)} | "
        info_linha1 += f"Total Processado: {self.total_deteccoes} | "
        info_linha1 += f"FPS: {fps:.1f}"
        
        info_linha2: str = f"Celulares: {contadores.get(67, 0)} | "
        info_linha2 += f"Laptops: {contadores.get(76, 0)} | "
        info_linha2 += f"Status: {status}"
        
        info_linha3: str = f"Threshold: {self.conf_threshold:.2f} | "
        info_linha3 += f"Frames: {self.frames_processados}"
        
        # Desenho das informações no painel
        font: int = cv2.FONT_HERSHEY_SIMPLEX
        font_scale: float = 0.5
        cor_texto: ColorBGR = (255, 255, 255)
        
        cv2.putText(painel, info_linha1, (10, 25), font, font_scale, cor_texto, 1)
        cv2.putText(painel, info_linha2, (10, 50), font, font_scale, cor_texto, 1)
        cv2.putText(painel, info_linha3, (10, 75), font, font_scale, cor_texto, 1)
        
        # Combinação do painel com o frame principal
        frame_completo: np.ndarray = np.vstack([painel, frame])
        
        return frame_completo
    
    def executar_deteccao_tempo_real(
        self, 
        camera_id: int = 0, 
        salvar_video: bool = False
    ) -> bool:
        """
        Executa o sistema de detecção em tempo real.
        
        Args:
            camera_id: Identificador da câmera (0 para padrão)
            salvar_video: Se deve salvar o vídeo processado
            
        Returns:
            True se execução foi bem-sucedida, False caso contrário
        """
        self.logger.info(f"Iniciando detecção em tempo real - Camera {camera_id}")
        
        # Inicialização da captura de vídeo
        cap: cv2.VideoCapture = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            self.logger.error("Falha ao inicializar câmera")
            self._exibir_dicas_camera()
            return False
        
        # Configurações da câmera
        self._configurar_camera(cap)
        
        # Configuração de gravação de vídeo (opcional)
        video_writer: Optional[cv2.VideoWriter] = None
        if salvar_video:
            video_writer = self._configurar_gravacao_video()
        
        # Variáveis de controle
        controlador_sessao = ControladorSessao()
        
        # Criação de diretório para screenshots
        screenshots_dir = Path("capturas")
        screenshots_dir.mkdir(exist_ok=True)
        
        self._exibir_controles()
        
        try:
            while True:
                resultado_loop = self._processar_loop_principal(
                    cap, video_writer, controlador_sessao, screenshots_dir
                )
                
                if not resultado_loop:
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Sistema interrompido pelo usuário (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Erro durante execução: {e}")
        finally:
            self._finalizar_recursos(cap, video_writer)
            return True
    
    def _exibir_dicas_camera(self) -> None:
        """Exibe dicas para resolução de problemas de câmera."""
        self.logger.info("Verificações recomendadas:")
        self.logger.info("- Câmera conectada e funcionando")
        self.logger.info("- Permissions de acesso à câmera")
        self.logger.info("- Outros programas não estão usando a câmera")
    
    def _configurar_camera(self, cap: cv2.VideoCapture) -> None:
        """Configura parâmetros da câmera."""
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    def _configurar_gravacao_video(self) -> cv2.VideoWriter:
        """Configura gravação de vídeo."""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"deteccao_{timestamp}.avi"
        video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (1280, 820))
        self.logger.info(f"Gravação de vídeo ativada: {video_filename}")
        return video_writer
    
    def _exibir_controles(self) -> None:
        """Exibe controles disponíveis."""
        self.logger.info("Sistema de detecção iniciado")
        self.logger.info("Controles disponíveis:")
        self.logger.info("- Q/ESC: Encerrar sistema")
        self.logger.info("- S: Capturar screenshot")
        self.logger.info("- ESPAÇO: Pausar/Retomar")
        self.logger.info("- +/-: Ajustar threshold de confiança")
    
    def _processar_loop_principal(
        self,
        cap: cv2.VideoCapture,
        video_writer: Optional[cv2.VideoWriter],
        controlador: 'ControladorSessao',
        screenshots_dir: Path
    ) -> bool:
        """
        Processa o loop principal de detecção.
        
        Args:
            cap: Captura de vídeo
            video_writer: Gravador de vídeo (opcional)
            controlador: Controlador de sessão
            screenshots_dir: Diretório para screenshots
            
        Returns:
            True para continuar loop, False para encerrar
        """
        frame_atual: Optional[np.ndarray] = None
        deteccoes: List[DetectionDict] = []
        
        if not controlador.sistema_pausado:
            ret, frame = cap.read()
            if not ret:
                self.logger.error("Falha na leitura do frame")
                return False
            
            frame_atual = frame
            frame_processado, deteccoes = self.processar_frame(frame)
            controlador.atualizar_fps()
        else:
            if frame_atual is not None:
                frame_processado = frame_atual.copy()
            else:
                return True
        
        # Status do sistema
        status = "PAUSADO" if controlador.sistema_pausado else "ATIVO"
        
        # Desenho da interface
        frame_final = self._desenhar_interface_informacoes(
            frame_processado, deteccoes, controlador.fps_atual, status
        )
        
        # Gravação de vídeo (se ativada)
        if video_writer is not None and not controlador.sistema_pausado:
            video_writer.write(frame_final)
        
        # Exibição do frame
        cv2.imshow('Sistema de Detecção de Dispositivos', frame_final)
        
        # Processamento de comandos do teclado
        return self._processar_comandos_teclado(
            frame_processado, controlador, screenshots_dir
        )
    
    def _processar_comandos_teclado(
        self,
        frame: np.ndarray,
        controlador: 'ControladorSessao',
        screenshots_dir: Path
    ) -> bool:
        """
        Processa comandos do teclado.
        
        Args:
            frame: Frame atual para screenshot
            controlador: Controlador de sessão
            screenshots_dir: Diretório para screenshots
            
        Returns:
            True para continuar, False para encerrar
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q ou ESC
            self.logger.info("Encerrando sistema por comando do usuário")
            return False
            
        elif key == ord('s'):  # Screenshot
            self._salvar_screenshot(frame, screenshots_dir)
            
        elif key == ord(' '):  # Pausa/Resume
            controlador.alternar_pausa()
            status_msg = "pausado" if controlador.sistema_pausado else "retomado"
            self.logger.info(f"Sistema {status_msg}")
            
        elif key == ord('+') or key == ord('='):  # Aumentar threshold
            self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
            self.logger.info(f"Threshold ajustado para: {self.conf_threshold:.2f}")
            
        elif key == ord('-'):  # Diminuir threshold
            self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
            self.logger.info(f"Threshold ajustado para: {self.conf_threshold:.2f}")
        
        return True
    
    def _salvar_screenshot(self, frame: np.ndarray, screenshots_dir: Path) -> None:
        """Salva screenshot do frame atual."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = screenshots_dir / f"deteccao_{timestamp}.jpg"
        cv2.imwrite(str(screenshot_path), frame)
        self.logger.info(f"Screenshot salvo: {screenshot_path}")
    
    def _finalizar_recursos(
        self, 
        cap: cv2.VideoCapture, 
        video_writer: Optional[cv2.VideoWriter]
    ) -> None:
        """Finaliza e libera recursos."""
        cap.release()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Relatório final
        self.logger.info("=== RELATÓRIO DE SESSÃO ===")
        self.logger.info(f"Frames processados: {self.frames_processados}")
        self.logger.info(f"Total de detecções: {self.total_deteccoes}")
        self.logger.info("Recursos liberados com sucesso")


class ControladorSessao:
    """
    Classe para controlar estado da sessão de detecção.
    
    Attributes:
        sistema_pausado: Estado de pausa do sistema
        fps_counter: Contador para cálculo de FPS
        fps_inicio: Timestamp de início para FPS
        fps_atual: FPS atual calculado
    """
    
    def __init__(self) -> None:
        """Inicializa controlador de sessão."""
        self.sistema_pausado: bool = False
        self.fps_counter: int = 0
        self.fps_inicio: float = cv2.getTickCount()
        self.fps_atual: float = 0.0
    
    def alternar_pausa(self) -> None:
        """Alterna estado de pausa do sistema."""
        self.sistema_pausado = not self.sistema_pausado
    
    def atualizar_fps(self) -> None:
        """Atualiza cálculo de FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            fps_fim = cv2.getTickCount()
            self.fps_atual = 30.0 / ((fps_fim - self.fps_inicio) / cv2.getTickFrequency())
            self.fps_inicio = fps_fim
            self.fps_counter = 0


def verificar_dependencias() -> bool:
    """
    Verifica se todas as dependências necessárias estão instaladas.
    
    Returns:
        True se todas as dependências estão disponíveis
    """
    dependencias_requeridas: Dict[str, str] = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'ultralytics': 'ultralytics'
    }
    
    dependencias_faltando: List[str] = []
    
    for modulo, pacote in dependencias_requeridas.items():
        try:
            __import__(modulo)
        except ImportError:
            dependencias_faltando.append(pacote)
    
    if dependencias_faltando:
        print("ERRO: Dependências faltando:")
        for pacote in dependencias_faltando:
            print(f"  - {pacote}")
        print("\nComando de instalação:")
        print(f"pip install {' '.join(dependencias_faltando)}")
        return False
    
    return True


def main() -> None:
    """
    Função principal do sistema de detecção.
    
    Coordena a inicialização e execução do sistema completo,
    incluindo verificações de dependências e tratamento de erros.
    """
    print("=" * 60)
    print("SISTEMA DE DETECÇÃO DE DISPOSITIVOS MÓVEIS")
    print("Baseado em YOLO - Versão Corporativa")
    print("=" * 60)
    
    # Verificação de dependências
    if not verificar_dependencias():
        print("Sistema não pode ser iniciado devido a dependências faltando.")
        return
    
    try:
        # Inicialização do sistema
        detector = DetectorDispositivos(
            model_path='yolov8s.pt',  # Modelo balanceado para ambiente corporativo
            conf_threshold=0.5
        )
        
        # Execução do sistema principal
        sucesso = detector.executar_deteccao_tempo_real(
            camera_id=0,
            salvar_video=False  # Altere para True se desejar gravar
        )
        
        if not sucesso:
            print("Sistema encerrado com falhas.")
            sys.exit(1)
            
    except Exception as e:
        print(f"ERRO CRÍTICO: {e}")
        print("\nSoluções recomendadas:")
        print("1. Verificar conexão da câmera")
        print("2. Fechar outros programas que usam câmera")
        print("3. Executar como administrador")
        print("4. Verificar logs em 'detector_dispositivos.log'")
        sys.exit(1)


if __name__ == "__main__":
    main()