def process_image(image_path):
    try:
        # Inicializar MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("No se pudo cargar la imagen")

        # Convertir a RGB y escala de grises
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detectar puntos faciales
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise Exception("No se detectó ninguna cara en la imagen")

        height, width = gray_image.shape

        # Obtener puntos faciales
        keyfacial_df_copy = {'Image': [gray_image], 'Points': []}
        for landmark in results.multi_face_landmarks[0].landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            keyfacial_df_copy['Points'].append((x, y))

        # Función para crear imágenes con puntos faciales
        def annotate_image(image, points):
            plt.clf()
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(image, cmap='gray')
            for j in range(1, len(points), 2):
                plt.plot(points[j - 1][0], points[j - 1][1], 'rx')

            # Guardar imagen en memoria
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)

            return base64.b64encode(buf.getvalue()).decode('utf-8')

        # Procesar cada propiedad de la imagen
        original_annotated = annotate_image(keyfacial_df_copy['Image'][0], keyfacial_df_copy['Points'])
        flipped_image = cv2.flip(keyfacial_df_copy['Image'][0], 1)
        flipped_annotated = annotate_image(flipped_image, keyfacial_df_copy['Points'])
        bright_image = np.clip(random.uniform(1.5, 2) * keyfacial_df_copy['Image'][0], 0, 255)
        bright_annotated = annotate_image(bright_image, keyfacial_df_copy['Points'])

        return {
            'original': original_annotated,
            'flipped': flipped_annotated,
            'bright': bright_annotated
        }

    except Exception as e:
        print(f"Error en process_image: {str(e)}")
        raise
    finally:
        plt.close('all')
