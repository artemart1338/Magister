Магистерская работа по теме "Разработка системы идентификации рукописной подписи с помощью сверточных нейросетей"


----------------------------------------------------------------------------------------------------------------- - 22.01
Первая версия программы (Свой метод, отсутствие в методе градации серого, графиков и обрезки изображений) 

------------------------------------------------------- - 22.01
Отчёт содержит первый пункт, 20 готовых страниц 

----------------------------------------------------------------------------------------------------------------- - 25.01
Вторая версия магистерской работы
Версия всё так же с собственной архитектурой. Установлена conda и обучение происходит на ресурсах видеокарты.
Добавлена обработка изображений в разных размерах 128x128, 256x256,512x512. Добавлены графики обучения и валидации.
Модель требует усовершенствования в сторону точности.

Так же стоит добавить графики для первых двух челов(как в примере к магистерской работе).
Не ясно стоит ли обрабатывать изображения во время работы программы либо же изначально использовать обработанные изображения в разных размерах. Как это отражается на точности?
Вопрос к содержанию выборки - нужно ли использовать те же данные, на основе которых идёт обучение? Проще говоря - обучающая выборка должна быть такая же как и тестовая или как?
Насколько большая должна быть выборка?

Нужно довести до ума с помощью графиков первую версию и отослать с вопросами. Но тогда стоит в первой версии поиграть с разрешением. Но тогда вопрос к мощностям процессора, ведь будет обработка через проц.
